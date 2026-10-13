import yaml

from crysim.src import REG_ARGS, DEBUG
from crysim.src.registry import RegressionName
from crysim.src.regressor.poly_reg_nn import get_learning_model_name, QuadraticOptimization
from logzero import logger as log_printer


class TrainingSettingBase:
    def __init__(self, epochs, batch_size, lr, lr_s_name, warmup_steps, weight_decay, valid_rate, test_rate,
                 patience, mt_lambda, alpha, beta, ema_momentum, loss_type, y_scaling_method):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_s_name = lr_s_name
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.mt_lambda = mt_lambda
        self.alpha, self.beta = alpha, beta
        self.ema_momentum = ema_momentum
        self.loss_type = loss_type
        self.y_scaling_method = y_scaling_method
        self.X, self.y = None, None
        self.logger = None
        self.valid_rate, self.test_rate = valid_rate, test_rate
        self.patience = int(epochs / 2) if patience is None or patience == 'None' else int(patience)

        self.learn_name = f"{epochs}-{batch_size}-{lr}-{lr_s_name}-{warmup_steps}-{weight_decay}" \
                          f"-{mt_lambda}-{alpha}-{beta}-{ema_momentum}"
        return

    def unwrap(self):
        raise NotImplementedError

    def get_dataset(self, X, y):
        self.X, self.y = X, y
        return

    @classmethod
    def load(cls, *args):
        raise NotImplementedError


class RegressorProxy(TrainingSettingBase):
    def __init__(self, model, learning_model, interaction_terms, k_fm, valid_rate, test_rate,
                 epochs, batch_size, lr, lr_s_name, warmup_steps, weight_decay, patience,
                 mt_lambda, alpha, beta, ema_momentum, loss_type, y_scaling_method):
        super().__init__(epochs=epochs, batch_size=batch_size, lr=lr, lr_s_name=lr_s_name, warmup_steps=warmup_steps,
                         weight_decay=weight_decay, mt_lambda=mt_lambda, alpha=alpha, beta=beta, patience=patience,
                         ema_momentum=ema_momentum, loss_type=loss_type, y_scaling_method=y_scaling_method, 
                         valid_rate=valid_rate, test_rate=test_rate)
        self.model = model
        self.reg_name = get_learning_model_name(learning_model_order=learning_model,
                                                learning_model_type=model, k_fm=k_fm)

        self.learning_model = learning_model
        if (self.model == RegressionName.fm()[0]) and (self.learning_model == 2):
            self.learning_model = 'fm'
        self.interaction_terms, self.k_fm = interaction_terms, k_fm
        self.for_binary_model = None
        return

    def get_logger(self, logger):
        self.logger = logger
        self.logger.record_learning_settings(epochs=self.epochs, batch_size=self.batch_size,
                                             lr=self.lr, lr_s_name=self.lr_s_name,
                                             patience=self.patience,
                                             warmup_steps=self.warmup_steps,
                                             weight_decay=self.weight_decay,
                                             mt_lambda=self.mt_lambda,
                                             loss_type=self.loss_type,
                                             alpha=self.alpha, beta=self.beta,
                                             ema_momentum=self.ema_momentum,
                                             y_scaling_method=self.y_scaling_method)
        if (self.interaction_terms is None) or (self.interaction_terms == 0):
            log_printer.info("In objective functions, "
                                         "all bits are involved for calculating interactive terms")
        else:
            log_printer.info(f"In objective functions, only the last {self.interaction_terms} terms "
                                         f"participate in calculating interactive terms")
        return
    
    def is_regression_on_bits(self, for_binary_model):
        self.for_binary_model = for_binary_model
        return

    def unwrap(self):
        regressor = QuadraticOptimization(X=self.X, y=self.y,
                                          model=self.model, k_fm=self.k_fm, for_binary_model=self.for_binary_model,
                                          learning_model=self.learning_model, interaction_terms=self.interaction_terms,

                                          epochs=self.epochs, batch_size=self.batch_size, patience=self.patience,
                                          valid_rate=self.valid_rate, test_rate=self.test_rate,
                                          x_scaling_method=None, y_scaling_method=self.y_scaling_method,
                                          lr=self.lr, lr_s_name=self.lr_s_name, warmup_steps=self.warmup_steps,
                                          weight_decay=self.weight_decay, loss_type=self.loss_type,
                                          alpha=self.alpha, beta=self.beta, mt_lambda=self.mt_lambda,
                                          ema_momentum=self.ema_momentum, logger=self.logger)
        return regressor

    @classmethod
    def load(cls, learning_model_type, mt_lambda):
        with open(REG_ARGS, 'r') as stream:
            params = yaml.safe_load(stream)
        reg_arg = params[f"quadratic_reg_param"]
        split_rate = params["split_rate"]
        epoch = int(reg_arg['epochs']) if not DEBUG else 5
        reg = cls(model=learning_model_type,
                  learning_model=reg_arg['learning_model_order'],
                  interaction_terms=int(reg_arg['interaction_terms']),
                  k_fm=int(reg_arg['k_fm']),

                  epochs=epoch,
                  batch_size=int(reg_arg['batch_size']),
                  patience=reg_arg['patience'],
                #   valid_rate=float(split_rate['small_valid']), test_rate=float(split_rate['small_test']),
                  valid_rate=float(split_rate['large_valid']), test_rate=float(split_rate['large_test']),
                  lr=float(reg_arg['lr']), lr_s_name=reg_arg['lr_s_name'],
                  warmup_steps=int(reg_arg['warmup_steps']),
                  weight_decay=float(reg_arg['weight_decay']),
                  mt_lambda=mt_lambda,
                  alpha=float(reg_arg['alpha']), beta=float(reg_arg['beta']),
                  ema_momentum=float(reg_arg['ema_momentum']),
                  loss_type=reg_arg['loss_type'],
                  y_scaling_method=reg_arg['y_scaling_method'])
        return reg
