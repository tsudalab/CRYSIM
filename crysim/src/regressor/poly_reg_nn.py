from itertools import combinations
from logzero import logger as log_printer
import numpy as np
import torch
# from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import PolynomialFeatures

from crysim.src.registry import RegressionName, Component
from crysim.src.regressor.trainer import TrainingWorkflowFromArray


def learning_model_assertion(learning_model):
    assert (learning_model in ['linear', 'quadratic', 'fm']) or \
           ((isinstance(learning_model, int)) and learning_model > 0), \
        "Learning model should be polynomials"
    return


def get_learning_model_name(learning_model_order, learning_model_type, k_fm):
    if learning_model_type == RegressionName.quad()[0]:
        learning_model_name = RegressionName.simple_model(learning_model_order)
    elif learning_model_type == RegressionName.fm()[0]:
        learning_model_name = RegressionName.factorization_machine(learning_model_order, k_fm)
    else:
        raise NotImplementedError
    return learning_model_name


def use_interaction_terms(interaction_terms):
    if (interaction_terms is None) or (interaction_terms <= 0):
        return False
    else:
        return True


def get_interaction_terms(var_list, interaction_terms):
    """ Sometimes we only want the bits for atomic configurations to interact,
        i.e., excluding lattice, symmetry parts. """
    if isinstance(var_list, list):
        return var_list[-interaction_terms:]
    elif isinstance(var_list, np.ndarray) or isinstance(var_list, torch.Tensor):
        return var_list[..., -interaction_terms:]


def quad_list_to_q_mat(quad_coef_list, num_var, interaction_terms):
    q_mat = np.zeros((num_var, num_var))
    if not use_interaction_terms(interaction_terms):
        q_mat[np.triu_indices(num_var, 1)] = quad_coef_list  # not including diagonal
    else:
        i, j = np.triu_indices(num_var, 1 + num_var - interaction_terms)
        i += num_var - interaction_terms
        q_mat[i, j] = quad_coef_list
    return q_mat


def get_high_order_terms(x, o):
    """ In qubo, we only need interactive terms, such as x_1*x_2, as high-order terms,
        since (x_1)^2 == x_1. """
    var_combs = list(combinations(x.T, o))
    order_terms = [np.prod(var_comb, axis=0) for var_comb in var_combs]
    # order_terms = []
    # for i in x.T:
    #     for j in x.T:
    #         order_terms.append(np.sum(i * j))
    return order_terms


def transform_to_poly_for_q(x, order, interaction_terms):
    """ combine numerical values """
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
    """ combine variables (such as for hyperopt, amplify) """
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


def assemble_fm_weight(w_inter):
    w_inter = get_high_order_terms(w_inter.T, 2)
    w_inter = np.sum(np.vstack(w_inter).astype(float), axis=1)
    return w_inter


def build_fm(w_lin, w_inter, var, custom_sum):
    lin_terms = var @ w_lin
    inter_terms = custom_sum(
        (custom_sum(w_inter[i][k] * var[i] for i in range(w_inter.shape[0]))) ** 2
        - custom_sum(w_inter[i][k] ** 2 * var[i] ** 2 for i in range(w_inter.shape[0]))
        for k in range(w_inter.shape[1])
    ) / 2
    return lin_terms, inter_terms


class LinearRegressionNN(torch.nn.Module, Component):
    def __init__(self, emb_in, emb_out):
        torch.nn.Module.__init__(self)
        Component.__init__(self, name=RegressionName.quad()[0], real_name=RegressionName.quad()[1])
        self.linear = torch.nn.Linear(emb_in, emb_out, bias=True)
        return

    def forward(self, x):
        return self.linear(x)

    def extract_weight(self):
        w = self.linear.weight.detach().cpu().numpy().flatten()
        b = float(self.linear.bias.item()) if (self.linear.bias is not None) else 0.0
        return w, b


# based on https://github.com/tsudalab/bVAE-IM/blob/main/im/bVAE-IM.py
class TorchFM(torch.nn.Module, Component):
    def __init__(self, emb_in, k, interaction_terms=None):
        # n: size of binary features
        # k: size of latent features
        torch.nn.Module.__init__(self)
        Component.__init__(self, name=RegressionName.fm()[0], real_name=RegressionName.fm()[1])
        self.interaction_terms = interaction_terms
        if not use_interaction_terms(interaction_terms):
            interaction_terms = emb_in
        self.V = torch.nn.Parameter(torch.randn(interaction_terms, k), requires_grad=True)
        self.lin = torch.nn.Linear(emb_in, 1, bias=True)
        self.init_params()
        return

    def init_params(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        return

    def _factorization_machine_operations(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2
        out_inter = 0.5 * (out_1 - out_2)
        return out_inter

    def forward(self, x):
        if self.interaction_terms is not None:
            x_i = get_interaction_terms(x, self.interaction_terms)
            out_inter = self._factorization_machine_operations(x_i)
        else:
            out_inter = self._factorization_machine_operations(x)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        out = out.squeeze(dim=1)
        return out

    def extract_weight(self):
        lin_w = self.lin.weight.detach().cpu().numpy().flatten()
        inter_w = self.V.data.detach().cpu().numpy()
        b = float(self.lin.bias.item()) if (self.lin.bias is not None) else 0.0
        return [lin_w, inter_w], b


class QuadraticOptimization(TrainingWorkflowFromArray):
    def __init__(self, X, y, model, for_binary_model, learning_model, epochs, batch_size, patience,
                 lr, lr_s_name, warmup_steps, weight_decay, mt_lambda, alpha, beta, loss_type,
                 ema_momentum, interaction_terms, k_fm, logger, valid_rate, test_rate,
                 x_scaling_method, y_scaling_method):

        learning_model_assertion(learning_model)
        assert model in [RegressionName.quad()[0], RegressionName.fm()[0]]
        if model == RegressionName.fm()[0]:
            assert k_fm is not None

        self.model = model
        self.for_binary_model = for_binary_model
        self.learning_model = learning_model
        self.x_scaling_method = x_scaling_method
        if self.for_binary_model:
            assert self.x_scaling_method is None
        self.y_scaling_method = y_scaling_method

        super().__init__(X=X, y=y,
                         epochs=epochs, batch_size=batch_size, lr=lr, lr_s_name=lr_s_name,
                         warmup_steps=warmup_steps, weight_decay=weight_decay, patience=patience,
                         mt_lambda=mt_lambda, alpha=alpha, beta=beta, loss_type=loss_type,
                         ema_momentum=ema_momentum, logger=logger, valid_rate=valid_rate, test_rate=test_rate,
                         x_scaling_method=x_scaling_method, y_scaling_method=y_scaling_method)

        self.y_pred, self.y_real = None, None

        # model related settings
        self.k_fm = k_fm
        self.interaction_terms = interaction_terms
        
        self.X = self._combine_high_order_terms(self.X)
        if self.x_test0 is not None:
            self.x_test0 = self._combine_high_order_terms(self.x_test0)

        # initialize learning_method parameters
        self.w = None

        if not self.for_binary_model:
            log_printer.info("All high-order terms are contained for continuous solver.")
        else:
            log_printer.info("Only interactive high-order terms are contained for binary solver.")

        self.max_retrain = 3
        return

    def _combine_high_order_terms(self, x_input):
        if self.learning_model in [1, 'linear', 'fm']:
            return x_input
        learning_model = 2 if self.learning_model in ['quadratic'] else self.learning_model
        if not self.for_binary_model:
            quadratic_featurizer = PolynomialFeatures(degree=int(learning_model))
            if use_interaction_terms(self.interaction_terms):
                x_i = get_interaction_terms(x_input, self.interaction_terms)
                x = np.concatenate([x_input, quadratic_featurizer.fit_transform(x_i)], axis=1)
            else:
                x = quadratic_featurizer.fit_transform(x_input)
        else:
            # x_quadratic = transform_tp_ploy_for_q(x, 2)
            # high_order_terms_num = combinations_with_replacement(range(x.shape[1]), learning_model)
            # high_order_terms_num = len(list(high_order_terms_num))
            # x_quadratic = quadratic_featurizer.fit_transform(x)[:, -high_order_terms_num:]
            x = transform_to_poly_for_q(x_input, order=learning_model, interaction_terms=self.interaction_terms)
        return x
    
    def set_data(self, x, y):
        x = x.reshape(-1, self.d)
        x = self._combine_high_order_terms(x)
        self.X, self.y = x, y
        self.reset_cash_t()
        return
    
    # def add_data(self, x, y):
    #     x = x.reshape(-1, self.d)
    #     x = self._combine_high_order_terms(x)
    #     self.X = np.append(self.X, x, axis=0)
    #     self.y = np.append(self.y, y, axis=0)
    #     self.X, self.y, _ = filter_duplicated_embedded_struct(self.X, self.y)
    #     self.reset_cash_t()
    #     if self.logger is not None:
    #         self.logger.save_training_set(X=self.X, y=self.y)
    #     return

    # def transform_x(self):
    #     """ For optimization, the whole dataset is leveraged.
    #         Therefore, we do not split the training and validation set when transformation. """
    #     if self.t_x is not None:
    #         return
    #     if (self.learning_model == 1) or (self.learning_model == "linear"):
    #         self.t_x = self._transform_itself(self.X, scaling_method=self.x_scaling_method)
    #     elif self.learning_model == 'fm':
    #         self.t_x = self.X
    #     else:
    #         x = self._combine_high_order_terms()
    #         self.t_x = self._transform_itself(x, scaling_method=self.x_scaling_method)
    #     return

    # def transform_y(self):
    #     if self.t_y is not None:
    #         return
    #     self.t_y = self._transform_itself(self.y, scaling_method=self.y_scaling_method)
    #     return
    
    # def _load_dataset(self):
    #     self.transform_x()
    #     self.transform_y()
    #     self.split(X=self.t_x, y=self.t_y)
    #     self.build_dataset()
    #     # self.dl_train = self._combine_dl_dataset(self.x_train, self.y_train, valid=False)
    #     # self.dl_test = self._combine_dl_dataset(self.x_test, self.y_test, valid=True)
    #     return

    # def _learn_weights(self, model):
    #     self.load_trainer(model)
    #     if self.no_valid:
    #         self.fit(dl_train=self.dl_train, dl_valid=self.dl_test)
    #     else:
    #         self.fit(dl_train=self.dl_train, dl_valid=self.dl_valid)
    #     self.evaluate()
    #     return

    def learn_weights(self):

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

        # self.load_dataset()

        for _ in range(self.max_retrain):
            if self.model == RegressionName.quad()[0]:
                # model = LinearRegressionNN(emb_in=self.t_x.shape[1], emb_out=1)
                model = LinearRegressionNN(emb_in=self.X.shape[1], emb_out=1)
            elif self.model == RegressionName.fm()[0]:
                model = TorchFM(emb_in=self.X.shape[1], k=self.k_fm, interaction_terms=self.interaction_terms)
            else:
                raise NotImplementedError
            self._learn_weights(model=model)
            if self.trainer.successful_training_flag:
                break
        self.w = self.trainer.get_weight()
        self.evaluate_all()

        # if self.model == RegressionName.fm()[0]:
        #     import time
        #     t1 = time.process_time()
        #     w1 = assemble_fm_weight(w)
        #     t2 = time.process_time()
        #     print(f"time cost: {t2 - t1}")
        #     w2 = fm_objective_function(w, as_dict=False)
        #     t3 = time.process_time()
        #     print(f"time cost: {t3 - t2}")

        return

    def get_weight(self):
        return self.w

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
