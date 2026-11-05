import logging
import os
import numpy as np
import pandas as pd
import warnings

from typing import Dict, Union, Callable

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from pyace.const import *
from pyace.basis import BBasisConfiguration, ACEBBasisSet

from pyace.multispecies_basisextension import compute_bbasisset_train_mask, expand_trainable_parameters
from pyace.lossfuncspec import LossFunctionSpecification


class BackendConfig:
    def __init__(self, backend_config_dict: Dict):
        self.backend_config_dict = backend_config_dict
        self.validate()

    @property
    def evaluator_name(self):
        return self.backend_config_dict[BACKEND_EVALUATOR_KW]

    def __getitem__(self, item):
        return self.backend_config_dict[item]

    def __setitem__(self, key, value):
        self.backend_config_dict[key] = value

    def validate(self):
        evaluator = self.backend_config_dict.get(BACKEND_EVALUATOR_KW)
        if evaluator == TENSORPOT_EVAL:
            warnings.warn("backend evaluator '{}' is deprecated and now uses the PyTorch implementation, "
                          "please use '{}' instead".format(TENSORPOT_EVAL, TORCHPOT_EVAL), DeprecationWarning)
            log.warning("backend evaluator '{}' is deprecated and now uses the PyTorch implementation, "
                        "please use '{}' instead".format(TENSORPOT_EVAL, TORCHPOT_EVAL))
            self.backend_config_dict[BACKEND_EVALUATOR_KW] = normalize_evaluator_name(evaluator)

    def get(self, item, default_value=None):
        return self.backend_config_dict.get(item, default_value)


class FitBackendAdapter:

    def __init__(self, backend_config: Union[Dict, BackendConfig], loss_spec: LossFunctionSpecification = None,
                 fit_config: Dict = None, callback: Callable = None, fit_metrics_callback: Callable = None,
                 test_metrics_callback: Callable = None):
        if isinstance(backend_config, dict):
            self.backend_config = BackendConfig(backend_config)
        else:
            self.backend_config = backend_config
        self.callback = callback
        self.loss_spec = loss_spec
        self.fit_config = fit_config
        self.res_opt = None
        self.fitter = None
        self.metrics = None
        self.fit_metrics_callback = fit_metrics_callback
        self.test_metrics_callback = test_metrics_callback
        self._pool = None  # worker processes of a multi-GPU `torchpot` fit (torchpotential.distributed.DevicePool)

    def close(self):
        """Stop the worker processes of a multi-GPU fit, if any."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    @property
    def evaluator_name(self):
        return self.backend_config.evaluator_name

    def fit(self,
            bbasisconfig: BBasisConfiguration,
            dataframe: pd.DataFrame,
            loss_spec: LossFunctionSpecification = None,
            fit_config: Dict = None, callback: Callable = None,
            test_dataframe: pd.DataFrame = None
            ) -> BBasisConfiguration:
        if loss_spec is None:
            loss_spec = self.loss_spec
        else:
            self.loss_spec = loss_spec
        if fit_config is None:
            fit_config = self.fit_config

        if callback is not None:
            self.callback = callback

        trainable_parameters = fit_config.get("trainable_parameters", [])  # default value = [] -> ["ALL"]
        # convert fit_blocks to indices of the blocks to fit
        elements = ACEBBasisSet(bbasisconfig).elements_name
        trainable_parameters_dict = expand_trainable_parameters(elements=elements,
                                                                trainable_parameters=trainable_parameters)

        log.info("Trainable parameters: {}".format(trainable_parameters_dict))

        # save globally
        self.trainable_parameters_dict = trainable_parameters_dict
        self.bbasisconfig = bbasisconfig

        if self.backend_config.evaluator_name == TORCHPOT_EVAL:
            from torchpotential.utils.device import is_oom_error
            while True:
                oom_message = None
                try:
                    self.setup_torchpot(bbasisconfig, dataframe, loss_spec, trainable_parameters_dict)
                    fit_res = self.run_torchpot_fit(bbasisconfig, dataframe, loss_spec, fit_config,
                                                    trainable_parameters_dict,
                                                    test_dataframe=test_dataframe)
                    self.log_optimization_result()
                    self._release_pool()
                    return fit_res
                except Exception as e:
                    if not is_oom_error(e):
                        raise e
                    # keep only the message: the exception's traceback would keep the failed tensors alive
                    oom_message = str(e)

                log.error("{} errors encountered".format(oom_message))
                if self.backend_config.get(BACKEND_BATCH_SIZE_REDUCTION_KW, True):
                    batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
                    batch_size_reduction_factor = self.backend_config.get(BACKEND_BATCH_SIZE_REDUCTION_FACTOR_KW,
                                                                          1.618)  # default - golden ratio
                    new_batch_size = int(batch_size / batch_size_reduction_factor)
                    log.info("Decrease batch size (by factor {}): {} -> {}".format(batch_size_reduction_factor,
                                                                                   batch_size, new_batch_size))
                    if new_batch_size < 1:
                        log.error("New batch size is too small, stopping")
                        raise RuntimeError("No further batch size reduction is possible, stopping")

                    self.backend_config[BACKEND_BATCH_SIZE_KW] = new_batch_size
                    # check the latest version of potential, update bbasisconfig to restart
                    try:
                        log.info("Attempt to get last version of potential")
                        bbasisconfig = self.fitter.tensorpot.potential.get_updated_config()
                        log.info("Last version of potential is extracted")
                    except Exception as e:
                        log.error("Can not get last version of potential: {}".format(e))
                    self.fitter = None
                    self._release_gpu_memory()
                else:
                    log.error("Use `backend:batch_size_reduction` option for automatic batch size reduction")
                    raise RuntimeError(("{} errors encountered. " +
                                        "Consider using `backend::{}=true` option for automatic batch size reduction"
                                        ).format(oom_message, BACKEND_BATCH_SIZE_REDUCTION_KW))

        elif self.backend_config.evaluator_name == PYACE_EVAL:
            self.setup_pyace(bbasisconfig, dataframe, loss_spec, trainable_parameters_dict)
            fit_res = self.run_pyace_fit(bbasisconfig, dataframe, loss_spec, fit_config, trainable_parameters_dict)
            self.log_optimization_result()
            return fit_res
        else:
            raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))

    def log_optimization_result(self, res_opt=None):
        if res_opt is None:
            res_opt = self.res_opt
        try:
            log.info(
                "Optimization result(success={success}, status={status}, message={message}, nfev={nfev}, njev={njev})".format(
                    success=res_opt.success,
                    status=res_opt.status,
                    message=res_opt.message,
                    nfev=res_opt.nfev, njev=res_opt.njev
                ))
        except Exception as e:
            log.error("Optimization result: not available: " + str(e))

    def get_evaluator_version_dict(self):
        try:
            if self.backend_config.evaluator_name == TORCHPOT_EVAL:
                import torchpotential
                return {TORCHPOT_EVAL + "_version": torchpotential.__version__}
            elif self.backend_config.evaluator_name == PYACE_EVAL:
                from pyace import __version__, get_ace_evaluator_version
                return {PYACE_EVAL + "_version": __version__, "ace_evaluator_version": get_ace_evaluator_version()}
            else:
                raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))
        except Exception as e:
            log.error(e)
        return {}

    def _release_gpu_memory(self):
        """Drop the failed fit and give cached GPU memory back before retrying with smaller batches."""
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._release_pool()

    def _release_pool(self):
        """Free the batches held by the worker processes (they are set up again by the next fit)."""
        if self._pool is not None:
            try:
                self._pool.release()
            except Exception as e:
                log.error("Can not release the worker processes: {}".format(e))

    def setup_torchpot(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                       loss_spec: LossFunctionSpecification,
                       trainable_parameters_dict: Dict
                       ) -> BBasisConfiguration:
        from torchpotential.potentials.ace import ACE
        from torchpotential.torchpot import TorchPotential
        from torchpotential.fit import FitTorchPotential
        from torchpotential.utils.utilities import batching_data, init_gpu_config, get_devices
        from torchpotential.utils.device import set_num_threads
        from torchpotential.constants import (LOSS_TYPE, LOSS_FORCE_FACTOR, LOSS_ENERGY_FACTOR, L1_REG,
                                              L2_REG, AUX_LOSS_FACTOR)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            gpu_config = self.backend_config.get(BACKEND_GPU_CONFIG, None)
            gpu_config = init_gpu_config(gpu_config)
            devices = get_devices(gpu_config)
            num_threads = self.backend_config.get(BACKEND_NUM_THREADS_KW, None)
            if len(devices) > 1:
                # one process per device: share the cores (the environment variable takes precedence, as usual)
                env_threads = os.environ.get("TORCHPOTENTIAL_NUM_THREADS")
                num_threads = int(env_threads) if env_threads else (
                        num_threads or max(1, (os.cpu_count() or 1) // len(devices)))
                log.info("Fitting on {} devices: {}".format(len(devices), [str(d) for d in devices]))
            pool = self._get_device_pool(devices, num_threads, gpu_config.get("mem_limit", 0))
            set_num_threads(num_threads)
            batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
            log.info("Loss function specification: " + str(loss_spec))
            log.info("Batch size: {}".format(batch_size))
            batches = batching_data(dataframe, batch_size=batch_size)
            n_batches = len(batches)
            loss_force_factor = loss_spec.kappa
            has_smoothness = (np.array([loss_spec.w0_rad, loss_spec.w1_rad, loss_spec.w2_rad]) != 0).any()
            has_orthogonality = loss_spec.w_orth != 0
            aux_loss_factor = []
            if has_orthogonality:
                aux_loss_factor += [np.float64(loss_spec.w_orth) / n_batches]
            if has_smoothness:
                aux_loss_factor += [np.float64(loss_spec.w0_rad) / n_batches,
                                    np.float64(loss_spec.w1_rad) / n_batches,
                                    np.float64(loss_spec.w2_rad) / n_batches]
            loss_specs = {
                LOSS_TYPE: 'per-atom',
                LOSS_FORCE_FACTOR: loss_force_factor,
                LOSS_ENERGY_FACTOR: (1 - loss_force_factor),
                L1_REG: np.float64(loss_spec.L1_coeffs) / n_batches,
                L2_REG: np.float64(loss_spec.L2_coeffs) / n_batches
            }
            if aux_loss_factor:
                loss_specs[AUX_LOSS_FACTOR] = aux_loss_factor

            ace_potential = ACE(bbasisconfig, compute_orthogonality=has_orthogonality,
                                compute_smoothness=has_smoothness)
            torchpotential = TorchPotential(ace_potential, loss_specs=loss_specs)

            display_step = self.backend_config.get('display_step', 20)
            self.fitter = FitTorchPotential(torchpotential, display_step=display_step, pool=pool)
            # assign total_number_of_functions to fitter
            total_number_of_functions = bbasisconfig.total_number_of_functions
            self.fitter.nfuncs = total_number_of_functions

    def _get_device_pool(self, devices, num_threads, mem_limit):
        """Worker processes for a fit on several devices (reused while the setup is unchanged); None for one device."""
        if len(devices) < 2:
            self.close()
            return None
        if self._pool is not None and not self._pool.matches(devices, num_threads, mem_limit):
            self.close()
        if self._pool is None:
            from torchpotential.distributed import DevicePool
            self._pool = DevicePool(devices, num_threads=num_threads, mem_limit=mem_limit)
        return self._pool

    def run_torchpot_fit(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                          loss_spec: LossFunctionSpecification, fit_config: Dict,
                          trainable_parameters_dict: Dict,
                          test_dataframe: pd.DataFrame = None
                          ) -> BBasisConfiguration:

        with warnings.catch_warnings():
            # adapt call of self._callback(current_bbasisconfig) from FitTorchPotential.callback(coeffs)
            def adapted_callback(coeffs):
                new_config = self.fitter.tensorpot.potential.get_updated_config(updating_coefs=coeffs)
                self._callback(new_config)

            jacobian_factor = compute_bbasisset_train_mask(bbasisconfig, trainable_parameters_dict)
            if np.all(jacobian_factor):
                jacobian_factor = None  # default value - train all
            else:
                jacobian_factor = jacobian_factor.astype(float)
            batch_size = self.backend_config.get(BACKEND_BATCH_SIZE_KW, 10)
            fit_options = fit_config.get(FIT_OPTIONS_KW, None)
            self.fitter.fit(dataframe, test_df=test_dataframe, niter=fit_config[FIT_NITER_KW],
                            optimizer=fit_config[FIT_OPTIMIZER_KW],
                            batch_size=batch_size, jacobian_factor=jacobian_factor,
                            callback=adapted_callback,  # call adapted_callback to pass current_bbasisconfig
                            options=fit_options,
                            fit_metric_callback=self.fit_metrics_callback,
                            test_metric_callback=self.test_metrics_callback
                            )

            self.res_opt = self.fitter.res_opt
            new_config = self.fitter.tensorpot.potential.get_updated_config(updating_coefs=self.res_opt.x)
            return new_config

    def setup_pyace(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                    loss_spec: LossFunctionSpecification,
                    trainable_parameters_dict: Dict
                    ) -> BBasisConfiguration:
        from pyace.pyacefit import PyACEFit

        parallel_mode = self.backend_config.get(BACKEND_PARALLEL_MODE_KW) or "serial"
        batch_size = len(dataframe)

        log.info("Loss function specification: " + str(loss_spec))
        display_step = self.backend_config.get('display_step', 20)
        # TODO: consider loss_spec.w_orth
        self.fitter = PyACEFit(basis=bbasisconfig,
                               loss_spec=loss_spec,
                               executors_kw_args=dict(parallel_mode=parallel_mode,
                                                      batch_size=batch_size,
                                                      n_workers=self.backend_config.get(BACKEND_NWORKERS_KW, None)
                                                      ),
                               seed=42,
                               display_step=display_step, trainable_parameters=trainable_parameters_dict)

        # maxiter = fit_config.get(FIT_NITER_KW, 100)
        #
        # fit_options = fit_config.get(FIT_OPTIONS_KW, {})
        # options = {"maxiter": maxiter, "disp": True}
        # options.update(fit_options)

        # assign total_number_of_functions to fitter
        total_number_of_functions = bbasisconfig.total_number_of_functions
        self.fitter.nfuncs = total_number_of_functions

    def run_pyace_fit(self, bbasisconfig: BBasisConfiguration, dataframe: pd.DataFrame,
                      loss_spec: LossFunctionSpecification, fit_config: Dict,
                      trainable_parameters_dict: Dict,
                      test_dataframe: pd.DataFrame = None
                      ) -> BBasisConfiguration:
        maxiter = fit_config.get(FIT_NITER_KW, 100)
        fit_options = fit_config.get(FIT_OPTIONS_KW, {})
        options = {"maxiter": maxiter, "disp": True}
        options.update(fit_options)

        self.fitter.fit(structures_dataframe=dataframe, method=fit_config[FIT_OPTIMIZER_KW],
                        options=options,
                        callback=self._callback,
                        fit_metric_callback=self.fit_metrics_callback
                        )

        self.res_opt = self.fitter.res_opt
        new_bbasisconf = self.fitter.bbasis_opt.to_BBasisConfiguration()
        # bbasisconfig.set_all_coeffs(new_bbasisconf.get_all_coeffs())
        return new_bbasisconf

    def setup_backend_for_predict(self, bbasisconfig):
        if bbasisconfig is None:
            raise ValueError("`bbasisconfig` couldn't be None for FitAdapter.setup_backend_for_predict")
        log.info("Setting {} backend for predicting".format(self.backend_config.evaluator_name))
        if self.backend_config.evaluator_name == TORCHPOT_EVAL:
            from torchpotential.potentials.ace import ACE
            from torchpotential.torchpot import TorchPotential
            from torchpotential.fit import FitTorchPotential
            ace_pot = ACE(bbasisconfig)
            tp = TorchPotential(ace_pot)
            self.fitter = FitTorchPotential(tensorpot=tp, eager=True)
        elif self.backend_config.evaluator_name == PYACE_EVAL:
            from pyace import PyACEFit
            self.fitter = PyACEFit(bbasisconfig)
        else:
            raise ValueError('{0} is not a valid evaluator'.format(self.backend_config.evaluator_name))

    def predict(self, structures_dataframe=None, bbasisconfig=None):
        if self.fitter is None:
            self.setup_backend_for_predict(bbasisconfig)
        return self.fitter.predict(structures_dataframe)

    def compute_metrics(self, energy_col='energy_corrected',
                        nat_column='NUMBER_OF_ATOMS', force_col='forces'):
        results = {}
        prediction = self.predict()
        l1, l2, smth1, smth2, smth3 = self.fitter.get_reg_components()
        datadf = self.fitter.get_fitting_data()

        datadf[force_col] = datadf[force_col].apply(np.array)
        datadf['w_forces'] = datadf['w_forces'].apply(np.reshape, newshape=[-1, 1])
        de = prediction['energy_pred'] - datadf[energy_col]
        df = prediction['forces_pred'] - datadf[force_col]
        e_loss = float(np.sum(datadf['w_energy'] * de ** 2))
        f_loss = np.sum((datadf['w_forces'] * df ** 2).map(np.sum))

        mae_pae = np.mean(np.abs(de / datadf[nat_column]))
        mae_e = np.mean(np.abs(de))
        mae_f = np.mean(np.abs(df).map(np.mean))
        rmse_pae = np.sqrt(np.mean(de ** 2 / datadf[nat_column]))
        rmse_e = np.sqrt(np.mean(de ** 2))
        rmse_f = np.sqrt(np.mean((df ** 2).map(np.mean)))

        results['mae_pae'] = mae_pae
        results['mae_e'] = mae_e
        results['mae_f'] = mae_f
        results['rmse_pae'] = rmse_pae
        results['rmse_e'] = rmse_e
        results['rmse_f'] = rmse_f

        results['e_loss'] = e_loss
        results['f_loss'] = f_loss
        results['l1'] = l1
        results['l2'] = l2
        results['radial_smooth'] = [smth1, smth2, smth3]

        return results

    def print_detailed_metrics(self, title='Iteration:'):
        if self.fitter is not None:
            self.fitter.print_detailed_metrics(title=title)

    def print_extended_metrics(self, title='Iteration:'):
        if self.fitter is not None:
            self.fitter.print_extended_metrics(title=title)

    def _callback(self, current_bbasisconfig: BBasisConfiguration):
        if self.callback is not None:
            self.callback(current_bbasisconfig)

    @property
    def last_loss(self):
        if self.backend_config.evaluator_name == TORCHPOT_EVAL:
            return self.fitter.loss_history[-1]
        elif self.backend_config.evaluator_name == PYACE_EVAL:
            return self.fitter.last_loss

    @property
    def last_fit_metric_data(self):
        if self.fitter is not None:
            last_fit_metric_data = self.fitter.last_fit_metric_data
            if last_fit_metric_data is None:
                last_fit_metric_data = {}
            return last_fit_metric_data

    @last_fit_metric_data.setter
    def last_fit_metric_data(self, value):
        if self.fitter is not None:
            self.fitter.last_fit_metric_data = value

    @property
    def last_test_metric_data(self):
        if self.fitter is not None:
            last_test_metric_data = self.fitter.last_test_metric_data
            if last_test_metric_data is None:
                last_test_metric_data = {}
            return last_test_metric_data

    @last_test_metric_data.setter
    def last_test_metric_data(self, value):
        if self.fitter is not None:
            self.fitter.last_test_metric_data = value
