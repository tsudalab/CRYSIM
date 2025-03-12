import numpy as np
import random
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from itertools import combinations
from amplify import VariableGenerator, sum_poly

from src.regressor.lin_reg_nn import LinearRegressionNN, TorchFM, LinearRegressionTrainer, \
    get_interaction_terms, get_high_order_terms, pearson_correlation_coefficient, use_interaction_terms
from src.registry import RegressionName
from src.log_analysis import print_or_record

NUM_PROCESS = -1


def transform_to_poly_for_q(x, order, interaction_terms):
    terms = []
    if not use_interaction_terms(interaction_terms):
        for o in range(1, order + 1):
            order_terms = get_high_order_terms(x, o)
            terms += order_terms
    else:
        terms += get_high_order_terms(x, 1)
        for o in range(2, order + 1):
            x_i = get_interaction_terms(x, interaction_terms)
            order_terms = get_high_order_terms(x_i, o)
            terms += order_terms
    terms = np.vstack(terms).T
    return terms


def collect_poly_terms(var, order, interaction_terms):
    poly_terms = []
    if not use_interaction_terms(interaction_terms):
        for o in range(1, order + 1):
            var_combs = list(combinations(var, o))
            poly_terms += var_combs
    else:
        poly_terms += list(combinations(var, 1))
        for o in range(2, order + 1):
            var_inter = get_interaction_terms(var, interaction_terms)
            var_combs = list(combinations(var_inter, o))
            poly_terms += var_combs
    return poly_terms


def learning_model_assertion(learning_model):
    assert (learning_model in ['linear', 'quadratic', 'fm']) or \
           ((isinstance(learning_model, int)) and learning_model > 0), \
        "Learning model should be polynomials"
    return


def get_learning_model_name(learning_model_order, learning_model_type, k_fm):
    if learning_model_type == RegressionName.sim()[0]:
        learning_model_name = RegressionName.simple_model(learning_model_order)
    elif learning_model_type == RegressionName.fm()[0]:
        learning_model_name = RegressionName.factorization_machine(learning_model_order, k_fm)
    else:
        raise NotImplementedError
    return learning_model_name


def assemble_fm_weight(fm_weight_list):
    lin_w, inter_w = fm_weight_list
    inter_w = get_high_order_terms(inter_w.T, 2)
    inter_w = np.sum(np.vstack(inter_w).astype(np.float64), axis=1)
    return np.concatenate([lin_w, inter_w])


def build_fm_based_on_amplify(fm_weight_list, amp_var):
    w_lin, w_inter = fm_weight_list
    lin_terms = sum_poly(w_lin.shape[0], lambda ii: w_lin[ii] * amp_var[ii])
    inter_terms = sum_poly(
        w_inter.shape[1],
        lambda k: (
                (sum_poly(w_inter.shape[0], lambda ii: w_inter[ii][k] * amp_var[ii])) ** 2
                - sum_poly(w_inter.shape[0], lambda ii: w_inter[ii][k] ** 2 * amp_var[ii] ** 2)
        )
    ) / 2
    return lin_terms, inter_terms


def fm_objective_function(fm_weight_list, as_dict):
    w_lin, w_inter = fm_weight_list
    var = VariableGenerator()
    var = var.array("Binary", w_lin.shape[0])
    lin_terms, inter_terms = build_fm_based_on_amplify(fm_weight_list, var)
    if as_dict:
        return (lin_terms + inter_terms).as_dict()
    else:
        def sort_values_based_on_keys(_dict):
            _sorted_keys = sorted(_dict.keys())
            return [_dict[k] for k in _sorted_keys]
        w = np.hstack([np.array(sort_values_based_on_keys(lin_terms.as_dict())),
                       np.array(sort_values_based_on_keys(inter_terms.as_dict()))])
        return w


class QuadraticSimulation:
    """
    Simple regression + Metropolis Hastings modification
    Based on CONBQA : CONtinuous Black-box optimization with Quantum Annealing
    """

    def __init__(self, X, y, model, for_binary_model, learning_model, epochs, batch_size,
                 lr, lr_s_name, warmup_steps, weight_decay, alpha, beta, ema_momentum,
                 interaction_terms, k_fm, logger,
                 # scaling_method=None,
                 scaling_method="Standardization",
                 # regression_method="MetropolisHastings",
                 regression_method="MaximumPosterior"
                 ):

        learning_model_assertion(learning_model)
        assert model in [RegressionName.sim()[0], RegressionName.fm()[0]]
        if model == RegressionName.fm()[0]:
            assert k_fm is not None

        assert type(X) is np.ndarray
        assert X.ndim == 2
        self.num_initial_data, self.d = X.shape
        assert type(y) is np.ndarray
        assert y.ndim == 1
        assert self.num_initial_data == y.size
        assert type(regression_method) is str

        self.logger = logger

        self.X = X
        self.y = y
        self.t_x, self.t_y = None, None
        self.y_pred, self.y_real = None, None

        self.model = model
        self.for_binary_model = for_binary_model
        self.learning_model = learning_model
        self.scaling_method = scaling_method
        self.regression_method = regression_method

        # model related settings
        self.k_fm = k_fm
        self.interaction_terms = interaction_terms

        # training settings
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_s_name = lr_s_name
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.alpha, self.beta = alpha, beta
        self.ema_momentum = ema_momentum

        # initialize learning_method parameters
        self.w = None
        self.burnin_steps = None
        self.sigma = None
        self.rate = None
        self.max_step_width = None

        if not self.for_binary_model:
            print_or_record(self.logger, "All high-order terms are contained for continuous solver.")
        else:
            print_or_record(self.logger, "Only interactive high-order terms are contained for binary solver.")

        if self.regression_method == "MetropolisHastings":
            self.set_regression_method_to_MetropolisHastings()
        elif self.regression_method == "MaximumPosterior":
            pass
        else:
            raise Exception("Unexpected regression_method")

    def set_regression_method_to_MetropolisHastings(self, burnin_steps=1000,
                                                    sigma=None, rate=0,
                                                    max_step_width=0.1):
        self.regression_method = "MetropolisHastings"
        self.burnin_steps = burnin_steps
        self.sigma = sigma
        self.rate = rate
        self.max_step_width = max_step_width
        if sigma is None:
            self.sigma = np.std(self.transform_y())

    def transform_x(self):
        if (hasattr(self, 't_x')) and (self.t_x is not None):
            return self.t_x

        if (self.learning_model == 1) or (self.learning_model == "linear"):
            if not self.for_binary_model:
                scaler = StandardScaler()
                self.t_x = scaler.fit_transform(X=self.X)
            else:
                self.t_x = self.X
        else:
            learning_model = 2 if self.learning_model in ['quadratic', 'fm'] else self.learning_model
            if not self.for_binary_model:
                quadratic_featurizer = PolynomialFeatures(degree=int(learning_model))
                if use_interaction_terms(self.interaction_terms):
                    x_i = get_interaction_terms(self.X, self.interaction_terms)
                    x = np.concatenate([self.X, quadratic_featurizer.fit_transform(x_i)], axis=1)
                else:
                    x = quadratic_featurizer.fit_transform(self.X)
                scaler = StandardScaler()
                self.t_x = scaler.fit_transform(X=x)
            else:
                # x_quadratic = transform_tp_ploy_for_q(x, 2)
                # high_order_terms_num = combinations_with_replacement(range(x.shape[1]), learning_model)
                # high_order_terms_num = len(list(high_order_terms_num))
                # x_quadratic = quadratic_featurizer.fit_transform(x)[:, -high_order_terms_num:]
                self.t_x = transform_to_poly_for_q(self.X, order=learning_model,
                                                   interaction_terms=self.interaction_terms)
        return self.t_x

    def transform_y(self):
        if (hasattr(self, 't_y')) and (self.t_y is not None):
            return self.t_y

        if self.scaling_method == "MinMaxNormalization":
            self.t_y = (self.y - np.min(self.y)) \
                        / (np.max(self.y) - np.min(self.y))
        elif self.scaling_method == "Standardization":
            self.t_y = (self.y - np.mean(self.y)) \
                        / np.std(self.y)
        elif self.scaling_method is None:
            self.t_y = self.y
        else:
            raise Exception("Unexpected scaling_method")
        return self.t_y

    def learn_weights(self):
        if self.regression_method == "MaximumPosterior":
            self.w = self.fit()
        elif self.regression_method == "MetropolisHastings":
            self.w = self.fit()
            transformed_x = self.transform_x()
            transformed_y = self.transform_y()
            self.w = MetropolisHastings(
                transformed_x,
                transformed_y,
                self.w,
                self.burnin_steps,
                self.sigma,
                self.rate,
                self.max_step_width,
            )
        else:
            raise Exception("Unexpected regression_method")
        return

    def convert_maximization_of_learned_function_into_qubo(self):
        """ This is specifically for dimod.BQM.from_qubo(q)
            For other types of qubo solver, directly outputting weight using get_weight can
            prevent unnecessary loops. """
        Q_dict = dict()
        if (self.learning_model == 1) or (self.learning_model == "linear"):
            for i in range(self.num_var):
                Q_dict[i] = self.w[i]
        elif (self.learning_model == 2) or (self.learning_model == "quadratic") or \
             (self.learning_model == "fm"):
            for i in range(self.num_var + 1):
                for j in range(i, self.num_var + 1):
                    idx_sum = int(np.sum(np.arange(self.num_var + 1 - i, self.num_var + 1)))
                    Q_dict[i, j] = self.w[j + idx_sum]
        else:
            raise Exception("Unexpected learning_model")
        return Q_dict

    def get_weight(self):
        return self.w

    def add_data(self, x, y):
        self.X = np.append(self.X, x.reshape(-1, self.d), axis=0)
        self.y = np.append(self.y, y)
        self.t_x, self.t_y = None, None
        if self.logger is not None:
            self.logger.save_training_set(X=self.X, y=self.y)
        return

    @property
    def num_var(self):
        return self.d

    def fit(self):
        print_or_record(self.logger, f"For learning, currently we have {self.X.shape[0]} samples.")

        # regressor_quadratic = LinearRegression(fit_intercept=False)
        # regressor_quadratic = Lasso(fit_intercept=False)
        # regressor_quadratic.fit(x_poly, transformed_y)
        # import pdb
        # pdb.set_trace()
        # return regressor_quadratic.coef_

        # linear_model = LinearRegression(fit_intercept=False)
        # linear_model = Lasso(alpha=1e-10, fit_intercept=False)
        # linear_model.fit(x, transformed_y)
        # return linear_model.coef_

        if self.model == RegressionName.sim()[0]:
            transformed_x = self.transform_x()
            transformed_y = self.transform_y()
            model = LinearRegressionNN(emb_in=transformed_x.shape[1], emb_out=1)
            trainer = LinearRegressionTrainer(model, transformed_x, transformed_y,
                                              epochs=self.epochs, batch_size=self.batch_size,
                                              lr=self.lr, lr_s_name=self.lr_s_name,
                                              warmup_steps=self.warmup_steps,
                                              weight_decay=self.weight_decay,
                                              alpha=self.alpha, beta=self.beta,
                                              ema_momentum=self.ema_momentum,
                                              logger=self.logger)
        elif self.model == RegressionName.fm()[0]:
            transformed_y = self.transform_y()
            model = TorchFM(emb_in=self.X.shape[1], k=self.k_fm, interaction_terms=self.interaction_terms)
            trainer = LinearRegressionTrainer(model, self.X, transformed_y,
                                              epochs=self.epochs, batch_size=self.batch_size,
                                              lr=self.lr, lr_s_name=self.lr_s_name,
                                              warmup_steps=self.warmup_steps,
                                              weight_decay=self.weight_decay,
                                              alpha=self.alpha, beta=self.beta,
                                              ema_momentum=self.ema_momentum,
                                              logger=self.logger)
        else:
            raise NotImplementedError
        trainer.train()
        w = trainer.get_weight()

        self.y_pred, self.y_real = trainer.get_pred_and_real()
        rmse, pcc = trainer.calculate_metrics(y_pred=self.y_pred, y=self.y_real)
        if self.logger is not None:
            self.logger.record_pes_metric(rmse=rmse, pcc=pcc)
        else:
            print(f"For the whole dataset, rmse: {rmse}, pcc: {pcc}")

        # if self.model == RegressionName.fm()[0]:
        #     import time
        #     t1 = time.process_time()
        #     w1 = assemble_fm_weight(w)
        #     t2 = time.process_time()
        #     print(f"time cost: {t2 - t1}")
        #     w2 = fm_objective_function(w, as_dict=False)
        #     t3 = time.process_time()
        #     print(f"time cost: {t3 - t2}")
        return w


def MetropolisHastings(Phi, transformed_y, w,
                       burnin_steps, sigma, rate, max_step_width):
    num_weight = len(w)
    choices = random.choices(range(num_weight), k=burnin_steps)
    estimated_y_old = np.dot(Phi, w)
    log_likelihood_old = - (np.linalg.norm(estimated_y_old - transformed_y,
                                           ord=2) ** 2) / \
                         (2.0 * (sigma ** 2))
    for i in range(burnin_steps):
        j = choices[i]
        w_j_old = w[j]
        log_posterior_old = log_likelihood_old - rate * w_j_old
        if w_j_old > max_step_width:
            w_j_new = w_j_old + \
                      2.0 * max_step_width * np.random.rand() - max_step_width
            log_proposal_old_to_new = np.log(1.0 / (2.0 * max_step_width))
        elif w_j_old < 0:
            w_j_new = max_step_width * np.random.rand()
            log_proposal_old_to_new = np.log(1.0 / max_step_width)
        else:
            w_j_new = (max_step_width + w_j_old) * np.random.rand()
            log_proposal_old_to_new = np.log(1.0 / (max_step_width + w_j_old))
        estimated_y_new = estimated_y_old + (w_j_new - w_j_old) * Phi[:, j]
        log_likelihood_new = - (np.linalg.norm(estimated_y_new - transformed_y,
                                               ord=2) ** 2) / \
                             (2.0 * (sigma ** 2))
        log_posterior_new = log_likelihood_new - rate * w_j_new
        if w_j_new > max_step_width:
            log_proposal_new_to_old = np.log(1.0 / (2.0 * max_step_width))
        elif w_j_new < 0:
            log_proposal_new_to_old = np.log(1.0 / max_step_width)
        else:
            log_proposal_new_to_old = np.log(1.0 / (max_step_width + w_j_new))
        if (log_posterior_new + log_proposal_new_to_old -
            log_posterior_old - log_proposal_old_to_new) >= \
                0:
            w[j] = w_j_new
            estimated_y_old = estimated_y_new
            log_likelihood_old = log_likelihood_new
        elif (log_posterior_new + log_proposal_new_to_old -
              log_posterior_old - log_proposal_old_to_new) >= \
                np.log(np.random.rand()):
            w[j] = w_j_new
            estimated_y_old = estimated_y_new
            log_likelihood_old = log_likelihood_new
    return w
