from src.registry import RegressionName
from src.regressor.poly_sim import get_learning_model_name, QuadraticSimulation
from src.log_analysis import print_or_record


def load_regressor(learning_model_type, reg_arg):
    reg = RegressorProxy(model=learning_model_type,
                         learning_model=reg_arg['learning_model_order'],
                         epochs=int(reg_arg['epochs']),
                         batch_size=int(reg_arg['batch_size']),
                         lr=float(reg_arg['lr']), lr_s_name=reg_arg['lr_s_name'],
                         warmup_steps=int(reg_arg['warmup_steps']),
                         weight_decay=float(reg_arg['weight_decay']),
                         alpha=float(reg_arg['alpha']), beta=float(reg_arg['beta']),
                         ema_momentum=float(reg_arg['ema_momentum']),
                         interaction_terms=int(reg_arg['interaction_terms']),
                         k_fm=int(reg_arg['k_fm']),
                         scaling_method="Standardization",
                         regression_method="MaximumPosterior")
    return reg


class RegressorProxy:
    def __init__(self, model, learning_model, epochs, batch_size,
                 lr, lr_s_name, warmup_steps, weight_decay, alpha, beta, ema_momentum,
                 interaction_terms, k_fm, scaling_method, regression_method):
        self.model = model
        self.learning_model = learning_model
        if (self.model == RegressionName.fm()[0]) and (self.learning_model == 2):
            self.learning_model = 'fm'

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_s_name = lr_s_name
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.alpha, self.beta = alpha, beta
        self.ema_momentum = ema_momentum

        self.interaction_terms, self.k_fm = interaction_terms, k_fm
        self.scaling_method = scaling_method
        self.regression_method = regression_method

        self.X, self.y = None, None
        self.for_binary_model = None
        self.logger = None

        self.reg_name = get_learning_model_name(learning_model_order=learning_model,
                                                learning_model_type=model, k_fm=k_fm)
        self.learn_name = f"{epochs}-{batch_size}-{lr}-{lr_s_name}-{warmup_steps}-{weight_decay}" \
                          f"-{alpha}-{beta}-{ema_momentum}"
        return

    def get_logger(self, logger):
        self.logger = logger
        self.logger.record_learning_settings(epochs=self.epochs, batch_size=self.batch_size,
                                             lr=self.lr, lr_s_name=self.lr_s_name,
                                             warmup_steps=self.warmup_steps,
                                             weight_decay=self.weight_decay,
                                             alpha=self.alpha, beta=self.beta,
                                             ema_momentum=self.ema_momentum)
        if (self.interaction_terms is None) or (self.interaction_terms == 0):
            print_or_record(self.logger, "In objective functions, "
                                         "all bits are involved for calculating interactive terms")
        else:
            print_or_record(self.logger, f"In objective functions, only the last {self.interaction_terms} terms "
                                         f"participate in calculating interactive terms")
        return

    def get_binary_type(self, for_binary_model):
        self.for_binary_model = for_binary_model
        return

    def get_init_dataset(self, X, y):
        self.X, self.y = X, y
        return

    def get_param(self, X, y, for_binary_model, logger):
        self.get_init_dataset(X, y)
        self.get_binary_type(for_binary_model)
        self.get_logger(logger)
        return

    def unwrap(self):
        regressor = QuadraticSimulation(X=self.X, y=self.y,
                                        epochs=self.epochs, batch_size=self.batch_size,
                                        model=self.model, for_binary_model=self.for_binary_model,
                                        learning_model=self.learning_model,
                                        scaling_method=self.scaling_method,
                                        lr=self.lr, lr_s_name=self.lr_s_name, warmup_steps=self.warmup_steps,
                                        weight_decay=self.weight_decay,
                                        alpha=self.alpha, beta=self.beta, ema_momentum=self.ema_momentum,
                                        interaction_terms=self.interaction_terms, k_fm=self.k_fm,
                                        regression_method=self.regression_method,
                                        logger=self.logger)
        return regressor
