from src.struct_utils import StructReader
from src.log_analysis import Logger, Logger4CSP
from src.registry import Component, OptimizerName


class ExtEstimator(Component):
    def __init__(self, name, real_name, system_name, e_cal, seed, n_steps, log_time):
        super().__init__(name=name, real_name=real_name)
        self.e_cal = e_cal

        self._load_output_name()
        self.logger = Logger(model_name=self.output_name, system_name=system_name, seed=seed,
                             require_time=log_time, n_steps=n_steps)
        self.logger.record_dataset_settings(init_training_size=n_steps, iter_sample_num=0, fail_num=None)
        self.logger4csp = Logger4CSP(self.logger)

    def _load_output_name(self):
        self.output_name = f'{self.name}-{self.e_cal.name}'
        self.real_output_name = f'{self.real_name} + {self.e_cal.real_name}'
        return

    def _estimate(self, init_struct, step):
        self.logger4csp.save_init_struct(init_struct, idx=step, idx_off=0)
        self.e_cal.build_atoms(init_struct)
        try:
            relaxed_struct, energy = self.e_cal.cal_relax()
        except RuntimeError:
            relaxed_struct, energy = init_struct, 999
        self.logger4csp.record_success_step(step, relaxed_struct, energy, None)
        return relaxed_struct, energy

    def estimate(self):
        raise NotImplementedError

    def __call__(self):
        self.estimate()
        return self.logger4csp.finish_logging()


class CryspyEstimator(ExtEstimator):
    def __init__(self, system_name, struct_starting_id, estimate_num, e_cal, cryspy_gen_file=None):
        super().__init__(name=OptimizerName.cryspy_rs()[0], real_name=OptimizerName.cryspy_rs()[1],
                         system_name=system_name, e_cal=e_cal, n_steps=estimate_num,
                         seed=struct_starting_id, log_time=True)
        self.struct_starting_id = struct_starting_id
        self.estimate_num = estimate_num

        if cryspy_gen_file is None:
            cryspy_gen_file = f"cryspy_file_{system_name}/data/init_POSCARS"
        struct_reader = StructReader(cryspy_gen_file)
        self.struct_list = struct_reader.to_struct()
        return

    def estimate(self):
        for i in range(self.struct_starting_id, self.struct_starting_id + self.estimate_num):
            init_struct = self.struct_list[i]
            self._estimate(init_struct, step=i)
        return
