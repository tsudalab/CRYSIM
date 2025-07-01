from itertools import combinations
import torch
# from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from src.regressor.lr_sche import LRSche, WarmupSche


try:
    torch.backends.cuda.preferred_linalg_library("magma")
except RuntimeError:
    pass
if torch.cuda.is_available():
    generator = torch.Generator(device='cuda')
else:
    generator = torch.Generator(device='cpu')


def use_interaction_terms(interaction_terms):
    if (interaction_terms is None) or (interaction_terms <= 0):
        return False
    else:
        return True


def get_interaction_terms(var_list, interaction_terms):
    if isinstance(var_list, list):
        return var_list[-interaction_terms:]
    elif isinstance(var_list, np.ndarray) or isinstance(var_list, torch.Tensor):
        return var_list[..., -interaction_terms:]


def get_high_order_terms(x, o):
    var_combs = list(combinations(x.T, o))
    order_terms = [np.prod(var_comb, axis=0) for var_comb in var_combs]
    # order_terms = []
    # for i in x.T:
    #     for j in x.T:
    #         order_terms.append(np.sum(i * j))
    return order_terms


def pearson_correlation_coefficient(x, y):
    miu_x, miu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x), np.std(y)
    cov = np.mean((x - miu_x) * (y - miu_y))
    pcc = cov / (sigma_x * sigma_y + 1e-6)
    return pcc


class LinearRegressionNN(torch.nn.Module):
    def __init__(self, emb_in, emb_out):
        super(LinearRegressionNN, self).__init__()
        self.linear = torch.nn.Linear(emb_in, emb_out, bias=False)
        return

    def forward(self, x):
        return self.linear(x)

    def extract_weight(self):
        return self.linear.weight.detach().cpu().numpy().flatten()


# based on https://github.com/tsudalab/bVAE-IM/blob/main/im/bVAE-IM.py
class TorchFM(torch.nn.Module):
    def __init__(self, emb_in, k, interaction_terms=None):
        # n: size of binary features
        # k: size of latent features
        super().__init__()
        self.interaction_terms = interaction_terms
        if not use_interaction_terms(interaction_terms):
            interaction_terms = emb_in
        self.V = torch.nn.Parameter(torch.randn(interaction_terms, k), requires_grad=True)
        self.lin = torch.nn.Linear(emb_in, 1, bias=False)
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
        return [lin_w, inter_w]


class MSEPCCLoss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        return

    def forward(self, x, y):
        mse = torch.mean(torch.pow(x - y, 2))
        miu_x, miu_y = torch.mean(x), torch.mean(y)
        sigma_x, sigma_y = torch.std(x), torch.std(y)
        cov = torch.mean((x - miu_x) * (y - miu_y))
        pcc = cov / (sigma_x * sigma_y + 1e-6)
        return self.alpha * mse - self.beta * torch.abs(pcc)


class LinearRegressionTrainer:
    def __init__(self, model, x, y, lr=1e-4, lr_s_name="None", warmup_steps=500,
                 weight_decay=1e-8, alpha=1, beta=1, ema_momentum=1,
                 epochs=1500, batch_size=10, patience=None, logger=None):
        self.model = model
        self.x = x
        self.y = y

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = int(epochs / 2) if patience is None else int(patience)

        self.lr = lr
        self.lr_s_name = lr_s_name
        self.warmup_steps = warmup_steps

        self.weight_decay = weight_decay
        # self.criterion = torch.nn.MSELoss()
        self.criterion = MSEPCCLoss(alpha=alpha, beta=beta)
        self.ema_momentum = ema_momentum
        if ema_momentum < 1:
            self.ema_recorder = EMARecorder(model=self.model, momentum=ema_momentum)

        self.logger = logger
        self.w_final = None
        return

    def train(self):
        self._load_dataset()
        # scaler = GradScaler()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        valid_loss_min = 10e7
        lr_scheduler = LRSche(optimizer=optimizer, epoch=self.epochs, start_lr=self.lr)
        lr_s = lr_scheduler(self.lr_s_name)
        warmup_scheduler = WarmupSche(optimizer=optimizer, warmup_steps=self.warmup_steps)
        warmup_s = warmup_scheduler()
        current_step = 0
        patience_epoch = 0
        # lr_list = []

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            tr_loss = 0
            patience_epoch += 1
            for train_data in self.dl_train:
                x, y = train_data[:, :-1], train_data[:, -1]
                if len(y) < np.max([(int(0.1 * self.batch_size)), 5]):
                    continue
                current_step += 1
                # with autocast():
                y_pred = self.model(x)
                loss = self.criterion(y_pred.flatten(), y)
                # if np.isnan(loss.detach().numpy()):
                #     print('nan')
                tr_loss += loss * len(y)
                # if torch.cuda.is_available():
                #     loss = loss.to('cuda:0')
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
                optimizer.step()
                if current_step <= self.warmup_steps:
                    warmup_s.step()
                if current_step == self.warmup_steps:
                    print(f"Finish warming up after {self.warmup_steps} steps.")

            # mixing previous network
            if self.ema_momentum < 1:
                self.ema_recorder.update_and_apply(self.model)

            # validation
            y_valid_pred, y_valid, loss_valid = self._validation()

            # learning rate decay
            if current_step > self.warmup_steps:
                if self.lr_s_name == "None":
                    pass
                elif self.lr_s_name == "ReduceLROnPlateau":
                    lr_s.step(metrics=loss_valid)
                else:
                    lr_s.step()

                # if self.lr_s_name not in ["None", "ReduceLROnPlateau"]:
                #     lr_list.append(lr_s.get_last_lr()[0])

            # save & print
            if loss_valid < valid_loss_min:
                valid_loss_min = loss_valid
                patience_epoch = 0
                self._save_weight()
                # if self.logger is not None:
                #     with open(self.logger.log_model_dir + '/' + f'valid_results{len(self.y)}.pkl', 'wb') as f:
                #         pickle.dump({'y_valid': y_valid, 'y_valid_pred': y_valid_pred}, f)

            # if (epoch == 0) or ((epoch + 1) % 10 == 0):
            #     print(f"Epoch {epoch + 1}, Training loss: {tr_loss / len(self.y_train)}")
            if (epoch == 0) or ((epoch + 1) % 100 == 0):
                print(f'Finish the {epoch + 1}-th iteration with loss {loss_valid}, '
                      f'minimum loss {valid_loss_min}.')

            # patience
            if patience_epoch > self.patience:
                break

        # if len(lr_list) != 0:
        #     d = Draw2d(x=list(range(len(lr_list))), y=lr_list, xlabel='epoch', ylabel='lr',
        #                pic_name=f'{self.lr_s_name}', save_dir='.')
        #     d.plot()
        return

    def _load_dataset(self):
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
            self.x, self.y, test_size=0.1, shuffle=True, random_state=42)
        dl_train = torch.hstack([torch.tensor(self.x_train, dtype=torch.float),
                                 torch.tensor(self.y_train.reshape(-1, 1), dtype=torch.float)])
        dl_valid = torch.hstack([torch.tensor(self.x_valid, dtype=torch.float),
                                 torch.tensor(self.y_valid.reshape(-1, 1), dtype=torch.float)])
        self.dl_train = DataLoader(dl_train, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.dl_valid = DataLoader(dl_valid, batch_size=len(self.y_valid), shuffle=False, generator=generator)
        return

    @staticmethod
    def _tensor2array(tensor):
        return tensor.flatten().detach().cpu().numpy()

    def _validation(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.dl_valid:
                x_valid, y_valid = data[:, :-1], data[:, -1]
            y_valid_pred = self.model(x_valid)
        loss_valid = self.criterion(y_valid_pred.flatten(), y_valid)
        return self._tensor2array(y_valid_pred), self._tensor2array(y_valid), \
            loss_valid.detach().cpu().numpy() / len(y_valid)

    def _save_weight(self):
        self.w_final = self.model.extract_weight()
        if self.logger is not None:
            # torch.save(self.model.state_dict(), self.logger.log_model_dir + '/' + f'model{len(self.y)}.pt')
            torch.save(self.model.state_dict(), self.logger.log_model_dir + '/' + f'model.pt')
        return

    def get_weight(self):
        return self.w_final

    def get_pred_and_real(self):
        self.model.eval()
        y_list, y_pred_list = [], []
        with torch.no_grad():
            for train_data in self.dl_train:
                x_tr, y_tr = train_data[:, :-1], train_data[:, -1]
                y_pred_tr = self.model(x_tr)
                y_list.append(self._tensor2array(y_tr))
                y_pred_list.append(self._tensor2array(y_pred_tr))
            for valid_data in self.dl_valid:
                x_v, y_v = valid_data[:, :-1], valid_data[:, -1]
                y_pred_v = self.model(x_v)
                y_list.append(self._tensor2array(y_v))
                y_pred_list.append(self._tensor2array(y_pred_v))
        return np.concatenate(y_pred_list), np.concatenate(y_list)

    @staticmethod
    def calculate_root_mean_squared_error(y_pred, y):
        return np.sqrt(np.mean((y_pred - y) ** 2)).round(5)

    @staticmethod
    def calculate_mean_absolute_error(y_pred, y):
        return np.mean(np.abs(y_pred - y)).round(5)

    @staticmethod
    def calculate_correlation_coefficient(y_pred, y):
        return pearson_correlation_coefficient(y_pred, y).round(5)

    def calculate_metrics(self, y_pred, y):
        return self. calculate_mean_absolute_error(y_pred, y), \
               self.calculate_root_mean_squared_error(y_pred, y), \
               self.calculate_correlation_coefficient(y_pred, y)


class EMARecorder:
    def __init__(self, model, momentum):
        self.recorder = {}
        self.momentum = momentum
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.recorder[name] = param.data.clone()
        return

    def update_and_apply(self, new_model):
        for name, param in new_model.named_parameters():
            if param.requires_grad:
                param.data = self.recorder[name] * (1 - self.momentum) + \
                             param.data.clone() * self.momentum
                self.recorder[name] = param.data
        return
