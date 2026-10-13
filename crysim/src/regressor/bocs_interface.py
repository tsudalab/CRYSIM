import torch
import yaml
from logzero import logger as log_printer

from crysim.src.registry import RegressionName
from crysim.src import REG_ARGS, PROJECT_ROOT
try:
    import sys
    sys.path.append(f"{PROJECT_ROOT}/bocs/src")
    from bocs.src.blr_batch_torch import BLR
    from bocs.src.bhs_pytorch import BayesianHorseshoeSampler
except ModuleNotFoundError:
    pass


class BOCSBase:
    def __init__(self, sample_time, lamda, y_standardization):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.sample_time = sample_time
        self.lamda = lamda
        self.y_standardization = y_standardization
        self.X, self.y = None, None
        self.w = None
        return
    
    def get_dataset(self, X, y):
        self.X, self.y = X, y
        return
    
    def get_logger(self, logger):
        self.logger = logger
        return
    
    def unwrap(self):
        self.bocs = None
        return
    
    def learn_weights(self):
        X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.y, dtype=torch.float32, device=self.device)
        self.bocs.update(X, y, standardize=self.y_standardization)
        self.w = []
        
        # from ex_bqp.py implementation
        cand_size_ts = max(1, self.sample_time - 1)
        ts_w_samples = self.bocs.sample_w(cand_size=cand_size_ts)
        for b_idx in range(self.sample_time):
            if b_idx == 0:
                # Slot 0: Greedy Exploitation (Posterior Mean Estimate)
                if hasattr(self.bocs, 'ws') and getattr(self.bocs, 'ws') is not None:
                    # Bayesian Horseshoe (BHS): predict uses mean of valid Gibbs samples
                    self.bocs.w_sample = self.bocs.ws.mean(dim=1).unsqueeze(0)
                else:
                    # BLR uses point estimate w
                    w_est = getattr(self.bocs, 'w', None)
                    self.bocs.w_sample = w_est.unsqueeze(0) if w_est is not None else None
            else:
                # Slots 1 to N: Thompson Sampling (Exploration)
                if ts_w_samples is not None and len(ts_w_samples) > 0:
                    idx = min(b_idx - 1, len(ts_w_samples) - 1)
                    self.bocs.w_sample = ts_w_samples[idx:idx+1]
            w = self.bocs.build_bqm().to_qubo()
            self.w.append(w)
        if self.sample_time == 1:
            self.w = self.w[0]
        return
    
    def get_weight(self):
        return self.w
    
    def set_data(self, x, y):
        self.X, self.y = x, y
        return
    
    @classmethod
    def load(cls):
        with open(REG_ARGS, 'r') as stream:
            params = yaml.safe_load(stream)
        reg_arg = params[f"bocs_reg_param"]
        reg = cls(sample_time=reg_arg['sample_time'],
                  sigma_noise=reg_arg['sigma_noise'],
                  w_var=reg_arg['w_var'],
                  lamda=reg_arg['lamda_for_l1'],
                  y_standardization=reg_arg['y_standardization'])
        return reg


class BOCSWrapper(BOCSBase):
    def __init__(self, sample_time, sigma_noise, w_var, lamda, y_standardization):
        super().__init__(sample_time=sample_time, lamda=lamda, y_standardization=y_standardization)
        self.sigma_noise, self.w_var = sigma_noise, w_var
        self.model = RegressionName.bocs()[0]
        self.reg_name = [f"{RegressionName.bocs()[0]}-{sample_time}", f"{RegressionName.bocs()[1]}-{sample_time}"]
        return

    def get_logger(self, logger):
        self.logger = logger
        log_printer.info('######### BOCS hyperparameters #########')
        log_printer.info(f'Sampling times (batch size of solutions): {self.sample_time}')
        log_printer.info(f'BLR sigma noise: {self.sigma_noise}')
        log_printer.info(f'Weight variance for BOCS normal prior: {self.w_var}')
        log_printer.info(f'Standardize labels: {self.y_standardization}')
        return
    
    def unwrap(self):
        self.bocs = BLR(r=self.X.shape[1], sigma_noise=self.sigma_noise, w_var=self.w_var, 
                        problem_type=1, device=self.device, lamda=self.lamda)
        return self

    @classmethod
    def load(cls):
        with open(REG_ARGS, 'r') as stream:
            params = yaml.safe_load(stream)
        reg_arg = params[f"{RegressionName.bocs()[0]}_reg_param"]
        reg = cls(sample_time=int(reg_arg['sample_time']),
                  sigma_noise=reg_arg['sigma_noise'],
                  w_var=reg_arg['w_var'],
                  lamda=reg_arg['lamda_for_l1'],
                  y_standardization=bool(reg_arg['y_standardization']))
        return reg


class BayesHSWrapper(BOCSBase):
    def __init__(self, sample_time, gibbs_itr, gibbs_burnin, gibbs_thin, lamda, y_standardization):
        super().__init__(sample_time=sample_time, lamda=lamda, y_standardization=y_standardization)
        self.gibbs_itr, self.gibbs_burnin, self.gibbs_thin = gibbs_itr, gibbs_burnin, gibbs_thin
        self.model = RegressionName.bhs()[0]
        self.reg_name = [f"{RegressionName.bhs()[0]}-{sample_time}", f"{RegressionName.bhs()[1]}-{sample_time}"]
        return
    
    def get_logger(self, logger):
        self.logger = logger
        log_printer.info('######### Bayesian Horseshoe Sampler hyperparameters #########')
        log_printer.info(f'Sampling times (batch size of solutions): {self.sample_time}')
        log_printer.info(f'Gibbs iterations for BOCS: {self.gibbs_itr}')
        log_printer.info(f'Gibbs burn-in: {self.gibbs_burnin}')
        log_printer.info(f'Gibbs thinning: {self.gibbs_thin}')
        log_printer.info(f'Standardize labels: {self.y_standardization}')
        return
    
    def unwrap(self):
        self.bocs = BayesianHorseshoeSampler(r=self.X.shape[1], nsamples=self.gibbs_itr, burnin=self.gibbs_burnin, thin=self.gibbs_thin, 
                                             problem_type=1, device=self.device, lamda=self.lamda)
        return self

    @classmethod
    def load(cls):
        with open(REG_ARGS, 'r') as stream:
            params = yaml.safe_load(stream)
        reg_arg = params[f"{RegressionName.bhs()[0]}_reg_param"]
        reg = cls(sample_time=int(reg_arg['sample_time']),
                  gibbs_itr=int(reg_arg['gibbs_itr']),
                  gibbs_burnin=int(reg_arg['gibbs_burnin']),
                  gibbs_thin=int(reg_arg['gibbs_thin']),
                  lamda=reg_arg['lamda_for_l1'],
                  y_standardization=bool(reg_arg['y_standardization']))
        return reg
