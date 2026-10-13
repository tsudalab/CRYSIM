import torch
from torch import nn
import numpy as np
from logzero import logger as log_printer
import time
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from crysim.src import DEFAULT_TORCH_DTYPE, DEBUG
from crysim.src.regressor.lr_sche import LRSche, WarmupSche

# try:
#     torch.backends.cuda.preferred_linalg_library('magma')
# except RuntimeError:
#     pass
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
generator = torch.Generator(device=device)


class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, x, y):
        miu_x, miu_y = torch.mean(x), torch.mean(y)
        sigma_x, sigma_y = torch.std(x), torch.std(y)
        cov = torch.mean((x - miu_x) * (y - miu_y))
        pcc = cov / (sigma_x * sigma_y + 1e-6)
        return - pcc


def get_loss_fn(loss_type):
    if loss_type in ['MSE', 'mse']:
        return nn.MSELoss()
    elif loss_type in ['MAE', 'mae']:
        return nn.L1Loss()
    elif loss_type in ['Huber', 'HUBER', 'huber']:
        return nn.HuberLoss()
    else:
        raise NotImplementedError


class RMSEPCCLoss(nn.Module):
    def __init__(self, alpha, beta, loss_type):
        assert loss_type in ['MSE', 'MAE', 'mse', 'mae'], "we support mse and mae loss"
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.loss_type = loss_type
        self.pcc = PCCLoss()
        self.l_loss = get_loss_fn(loss_type)
        return

    def forward(self, x, y):
        pcc = self.pcc(x, y)
        l_loss = self.l_loss(x, y)
        l_loss = torch.sqrt(l_loss) if self.loss_type in ['MSE', 'mse'] else l_loss
        return self.alpha * l_loss + self.beta * pcc


def pearson_correlation_coefficient(x, y):
    miu_x, miu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x), np.std(y)
    cov = np.mean((x - miu_x) * (y - miu_y))
    pcc = cov / (sigma_x * sigma_y + 1e-6)
    return pcc


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


class NNTrainer:
    def __init__(self, model, lr, lr_s_name, warmup_steps, weight_decay, mt_lambda, alpha, beta, loss_type, ema_momentum,
                 epochs, batch_size, patience, logger, check):
        self.model = model
        self.logger = logger
        t_p, a_p = print_trainable_parameters(model=self.model)
        log_printer.info(f"Current number of trainable parameters: {t_p}/{a_p}")
        self.successful_training_flag = False

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.lr = lr
        self.lr_s_name = lr_s_name
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        self.mt_lambda = mt_lambda
        self.alpha = alpha
        self.beta = beta
        self.loss_type = loss_type
        self._load_loss_fn()

        self.ema_momentum = ema_momentum
        if ema_momentum < 1:
            self.ema_recorder = EMARecorder(model=self.model, momentum=ema_momentum)

        self.w_final = None
        self.model_final = None
        self.check = check
        
        self.is_check_loss = False
        self.check_starting_epoch = 20
        self.check_failing_max_epoch = 100
        return

    def _load_loss_fn(self):
        self.loss_fn = {}
        if self.beta > 0:
            self.loss_fn['e'] = RMSEPCCLoss(alpha=self.alpha, beta=self.beta, loss_type=self.loss_type)
            self.loss_fn['ev'] = PCCLoss()
        else:
            self.loss_fn['e'] = get_loss_fn(self.loss_type)
            self.loss_fn['ev'] = get_loss_fn("MAE")
        if self.mt_lambda > 0:
            if self.beta > 0:
                self.loss_fn['f'] = RMSEPCCLoss(alpha=self.alpha, beta=self.beta, loss_type=self.loss_type)
                self.loss_fn['fv'] = PCCLoss()
            else:
                self.loss_fn['f'] = get_loss_fn(self.loss_type)
                self.loss_fn['fv'] = get_loss_fn("MAE")
        return

    @staticmethod
    def _tensor2array(tensor):
        if isinstance(tensor, dict):
            return np.vstack([tensor['e'].detach().cpu().numpy().flatten(),
                              tensor['f'].detach().cpu().numpy().flatten()]).T  # (idx, (e, f))
        else:
            new_tensor = tensor.detach().cpu().numpy()
            if new_tensor.size == len(new_tensor):
                new_tensor = new_tensor.flatten()
            return new_tensor

    def _unravel_samples(self, sample):
        # if self.model.name in graph_regressor:
        #     return sample.to(device), sample.y.to(device)
        # else:
        return sample[0].to(device), sample[1].to(device)

    def train(self, dl_train, dl_valid):
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
        epoch_print = max(int(self.epochs / 50), 1)
        valid_loss_min_0 = 10e7
        success_training_threshold = 10e7
        self.successful_training_flag = False

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            # tr_loss = 0
            patience_epoch += 1
            for train_data in dl_train:
                x, y = self._unravel_samples(train_data)
                if len(y) < np.max([int(0.1 * self.batch_size), 5]):
                    continue
                current_step += 1

                # with autocast():
                y_pred = self.model(x)
                if self.mt_lambda == 0 or not isinstance(y_pred, dict):
                    loss = self.loss_fn['e'](y_pred.flatten(), y)
                else:
                    loss = self.loss_fn['e'](y_pred['e'].flatten(), y[:, 0]) + \
                           self.mt_lambda * self.loss_fn['f'](y_pred['f'].flatten(), y[:, 1])

                # if np.isnan(loss.detach().numpy()):
                #     print('nan')
                # tr_loss += loss * len(y)
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
                    log_printer.info(f"Finish warming up after {self.warmup_steps} steps.")

            # mixing previous network
            if self.ema_momentum < 1:
                self.ema_recorder.update_and_apply(self.model)

            # validation
            y_valid_pred, y_valid, loss_valid = self._validation(dl_valid)

            # check training status
            if self.check:
                from draw_figure import DrawHistogram
                if epoch % 10 == 0:
                    y_pred_tr, y_tr = self.get_pred_and_real(dl_to_be_pred=dl_train, model=self.model)
                    y_pred_v, y_v = self.get_pred_and_real(dl_to_be_pred=dl_valid, model=self.model)
                    y_pred_tr = y_pred_tr[:, 0] if y_pred_tr.ndim == 2 and y_pred_tr.shape[1] > 1 else y_pred_tr
                    y_tr = y_tr[:, 0] if y_tr.ndim == 2 and y_tr.shape[1] > 1 else y_tr
                    y_pred_v = y_pred_v[:, 0] if y_pred_v.ndim == 2 and y_pred_v.shape[1] > 1 else y_pred_v
                    y_v = y_v[:, 0] if y_v.ndim == 2 and y_v.shape[1] > 1 else y_v
                    d = DrawHistogram(num_bar=100, data_list=[y_tr, y_v, y_pred_tr, y_pred_v],
                                      xlabel="Energy", ylabel="Number of points", pic_name=f"pred_{epoch}",
                                      lgd_name=["Training set", "Validation set",
                                                "Predicted training set", "Predicted validation set"])
                    d.draw()

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
                self._save_weight()
                if valid_loss_min - loss_valid > 0.001:
                    patience_epoch = 0
                valid_loss_min = loss_valid
                # if self.logger is not None:
                #     with open(self.logger.log_model_dir + '/' + f'valid_results{len(self.y)}.pkl', 'wb') as f:
                #         pickle.dump({'y_valid': y_valid, 'y_valid_pred': y_valid_pred}, f)

            # if (epoch == 0) or ((epoch + 1) % 10 == 0):
            #     print(f"Epoch {epoch + 1}, Training loss: {tr_loss / len(self.y_train)}")
            if (epoch == 0) or ((epoch + 1) % epoch_print == 0):
                log_printer.info(
                    f'{epoch + 1}-th epoch, training loss {loss.detach().cpu().numpy()}, valid loss {loss_valid}, min valid loss {valid_loss_min}.')
            
            if self.is_check_loss:
                if epoch == 0:
                    valid_loss_min_0 = valid_loss_min
                if epoch + 1 == self.check_starting_epoch:
                    success_training_threshold = valid_loss_min
                    log_printer.info(
                        f'After the {epoch + 1}-th epoch, record min valid loss {success_training_threshold}.')
                if epoch + 1 == self.check_starting_epoch + self.check_failing_max_epoch:
                    if np.abs(success_training_threshold - valid_loss_min) <= 0.1 and \
                        np.abs(valid_loss_min_0 - valid_loss_min) <= np.abs(valid_loss_min_0) * 0.25:
                        log_printer.info(
                            f'After the {epoch + 1}-th epoch, min valid loss {valid_loss_min} unchanged. Training failed. Please restart training.')
                        break
                    else:
                        log_printer.info(
                            f'After the {epoch + 1}-th epoch, min valid loss {valid_loss_min} updated. Trust the trained model.')
                        self.successful_training_flag = True
            else:
                self.successful_training_flag = True

            # patience
            if patience_epoch > self.patience:
                break

        # if len(lr_list) != 0:
        #     d = Draw2d(x=list(range(len(lr_list))), y=lr_list, xlabel='epoch', ylabel='lr',
        #                pic_name=f'{self.lr_s_name}', save_dir='.')
        #     d.plot()
        return

    def _validation(self, dl_valid):
        self.model.eval()
        y_pred_list, y_list = [], []
        with torch.no_grad():
            for data in dl_valid:
                x_valid, y_valid = self._unravel_samples(data)
                y_valid_pred = self.model(x_valid)
                y_pred_list.append(self._tensor2array(y_valid_pred))
                y_list.append(self._tensor2array(y_valid))
        y_valid_pred = np.vstack(y_pred_list) if y_pred_list[0].ndim == 2 else np.hstack(y_pred_list)
        y_valid = np.vstack(y_list) if y_list[0].ndim == 2 else np.hstack(y_list)
        y_valid_pred_t = torch.tensor(y_valid_pred, dtype=DEFAULT_TORCH_DTYPE, device=device)
        y_valid_t = torch.tensor(y_valid, dtype=DEFAULT_TORCH_DTYPE, device=device)
        if self.mt_lambda == 0 or not y_valid_pred_t.ndim == 2:
            loss_valid = self.loss_fn['ev'](y_valid_pred_t.flatten(), y_valid_t.flatten())
        else:
            loss_valid = self.loss_fn['ev'](y_valid_pred_t[:, 0].flatten(), y_valid_t[:, 0].flatten()) + \
                   self.mt_lambda * self.loss_fn['fv'](y_valid_pred_t[:, 1].flatten(), y_valid_t[:, 1].flatten())
        return y_valid_pred, y_valid, loss_valid.detach().cpu().numpy()

    def _save_weight(self):
        try:
            self.w_final = self.model.extract_weight()
        except AttributeError:
            pass
        self.model_final = deepcopy(self.model)
        if self.logger is not None:
            # torch.save(self.model.state_dict(), self.logger.log_model_dir + '/' + f'model{len(self.y)}.pt')
            torch.save(self.model.state_dict(), self.logger.log_model_dir + '/' + f'model{self.logger.new_struct_suc}.pt')
        return

    def get_weight(self):
        return self.w_final

    def get_pred_and_real(self, dl_to_be_pred, model=None):
        """ use the best model on validation instead of the latest model """
        model = self.model_final if model is None else model
        model.eval()
        y_list, y_pred_list = [], []
        with torch.no_grad():
            for data in dl_to_be_pred:
                x, y = self._unravel_samples(data)
                y_pred = model(x)
                y_list.append(self._tensor2array(y))
                y_pred_list.append(self._tensor2array(y_pred))
        if y_list[0].ndim == 1:
            return np.hstack(y_pred_list), np.hstack(y_list)
        else:
            return np.vstack(y_pred_list), np.vstack(y_list)
    
    def cleanup(self):
        if self.model_final is not None:
            del self.model_final
            self.model_final = None
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        return


class BarelyTrainNN:
    def __init__(self, X,
                 epochs, batch_size, lr, lr_s_name, warmup_steps, patience,
                 weight_decay, mt_lambda, alpha, beta, loss_type, ema_momentum, logger,
                 valid_rate, test_rate, y=None):

        self.logger = logger

        # dataset & splitting settings
        self.X = X
        self.y = y
        self.x_test0, self.y_test0 = None, None
        if isinstance(X, tuple):
            self.X, self.x_test0 = self.X
            if y is not None:
                self.y, self.y_test0 = self.y
            else:
                self.y, self.y_test0 = None, None
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = \
            None, None, None, None, None, None
        self.dl_train, self.dl_valid, self.dl_test = None, None, None
        self.split_seed = 0
        self.valid_rate, self.test_rate = valid_rate, test_rate
        self.no_valid = True if valid_rate <= 0 else False

        # training settings
        self.trainer = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_s_name = lr_s_name
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.mt_lambda = mt_lambda
        self.alpha, self.beta = alpha, beta
        self.loss_type = loss_type
        self.ema_momentum = ema_momentum
        self.patience = patience

        # predicted values
        self.y_pred_tr, self.y_tr, self.y_pred_v, self.y_v, self.y_pred_te, self.y_te = \
            None, None, None, None, None, None

        return

    @property
    def num_samples(self):
        if self.x_test0 is None:
            return len(self.X)
        else:
            return len(self.X) + len(self.x_test0)

    def _split_dataset(self, x, rate, y=None):
        if y is not None:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=rate, shuffle=True, random_state=self.split_seed)
            self.split_seed += 1
            return x_train, x_test, y_train, y_test
        else:
            ids = torch.arange(len(x))
            ids_train, ids_test = train_test_split(
                ids, test_size=rate, shuffle=True, random_state=self.split_seed)
            dl_train, dl_test = x[ids_train], x[ids_test]
            self.split_seed += 1
            return dl_train, dl_test, None, None

    def split(self, X=None, y=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        log_printer.info(f"For learning, currently we have {self.num_samples} samples")
        if self.x_test0 is None:
            self.x_train, self.x_test, self.y_train, self.y_test = self._split_dataset(self.X, rate=self.test_rate, y=self.y)
        else:
            log_printer.info(f"Use the input test set directly; input test rate ({self.test_rate}) is not used")
            self.x_train, self.x_test, self.y_train, self.y_test = self.X, np.array(self.x_test0), self.y, np.array(self.y_test0)
        if self.no_valid:
            log_printer.info(f"Training size: {len(self.x_train)}, test size: {len(self.x_test)}")
        else:
            self.x_train, self.x_valid, self.y_train, self.y_valid = self._split_dataset(self.x_train, rate=self.valid_rate, y=self.y_train)
            log_printer.info(f"Training size: {len(self.x_train)}, validation size: {len(self.x_valid)}, test size: {len(self.x_test)}")
        return

    def _combine_dl_dataset(self, x, y, valid):
        # dl_dataset = torch.hstack([torch.tensor(x, dtype=torch.float),
        #                            torch.tensor(y.reshape(-1, 1), dtype=torch.float)])
        dl_dataset = TensorDataset(torch.tensor(x, dtype=DEFAULT_TORCH_DTYPE),
                                   torch.tensor(y, dtype=DEFAULT_TORCH_DTYPE))
        if not valid:
            dl_dataset = DataLoader(
                dl_dataset, batch_size=self.batch_size, shuffle=True, generator=generator.manual_seed(self.split_seed))
        else:
            dl_dataset = DataLoader(
                dl_dataset, batch_size=len(y), shuffle=False, generator=generator.manual_seed(self.split_seed))
        return dl_dataset

    def build_dataset(self, x_list=None, y_list=None):
        if not self.no_valid:
            x_list = [self.x_train, self.x_valid, self.x_test] if x_list is None else x_list
            y_list = [self.y_train, self.y_valid, self.y_test] if y_list is None else y_list
        else:
            x_list = [self.x_train, self.x_test] if x_list is None else x_list
            y_list = [self.y_train, self.y_test] if y_list is None else y_list

        self.dl_train = self._combine_dl_dataset(x_list[0], y_list[0], valid=False)
        if not self.no_valid:
            self.dl_valid = self._combine_dl_dataset(x_list[1], y_list[1], valid=True)
            self.dl_test = self._combine_dl_dataset(x_list[2], y_list[2], valid=True)
        else:
            self.dl_test = self._combine_dl_dataset(x_list[1], y_list[1], valid=True)
        return

    def load_dataset(self):
        start = time.process_time()
        self._load_dataset()
        end = time.process_time()
        log_printer.info(f"loading dataset for training: {end - start}s")
        return

    def _load_dataset(self):
        self.split()
        self.build_dataset()
        return

    def load_trainer(self, model):
        self.trainer = NNTrainer(model,
                                 epochs=self.epochs, batch_size=self.batch_size,
                                 lr=self.lr, lr_s_name=self.lr_s_name,
                                 warmup_steps=self.warmup_steps, weight_decay=self.weight_decay,
                                 mt_lambda=self.mt_lambda, alpha=self.alpha, beta=self.beta, loss_type=self.loss_type,
                                 ema_momentum=self.ema_momentum, logger=self.logger,
                                 patience=self.patience, check=False)
        return self.trainer

    def fit(self, dl_train, dl_valid):
        self.trainer.train(dl_train=dl_train, dl_valid=dl_valid)
        return

    def _evaluate(self, valid_type, dl_to_be_evaluate=None):
        assert valid_type in ['training', 'validation', 'test']
        if dl_to_be_evaluate is None:
            if valid_type == 'training':
                dl_to_be_evaluate = self.dl_train
            elif valid_type == 'validation':
                dl_to_be_evaluate = self.dl_valid
            else:
                dl_to_be_evaluate = self.dl_test
        y_pred, y_real = self.trainer.get_pred_and_real(dl_to_be_evaluate)
        mae, rmse, pcc = self.calculate_metrics(y_pred=y_pred, y=y_real)
        if self.logger is not None:
            self.logger.record_valid_metric(mae=mae, rmse=rmse, pcc=pcc, valid_type=valid_type)
        else:
            log_printer.info(f"For the {valid_type} dataset, mae: {mae}, rmse: {rmse}, pcc: {pcc}")
        return y_pred, y_real

    def evaluate(self):
        if self.no_valid:
            self.y_pred_tr, self.y_tr = self._evaluate(valid_type='training')
            self.y_pred_te, self.y_te = self._evaluate(valid_type='test')
        else:
            self.y_pred_tr, self.y_tr = self._evaluate(valid_type='training')
            self.y_pred_v, self.y_v = self._evaluate(valid_type='validation')
            self.y_pred_te, self.y_te = self._evaluate(valid_type='test')
        return

    def _learn_weights(self, model):
        self.load_dataset()
        self.load_trainer(model)
        if self.no_valid:
            self.fit(dl_train=self.dl_train, dl_valid=self.dl_test)
        else:
            self.fit(dl_train=self.dl_train, dl_valid=self.dl_valid)
        self.evaluate()
        return

    def learn_weights(self):
        raise NotImplementedError

    def reset_cash(self):
        # Explicitly delete old DataLoaders and their underlying GPU tensors
        for dl_name in ['dl_train', 'dl_valid', 'dl_test']:
            dl = getattr(self, dl_name, None)
            if dl is not None:
                # Delete the TensorDataset inside the DataLoader
                if hasattr(dl, 'dataset') and dl.dataset is not None:
                    # TensorDataset holds GPU tensors - delete them
                    ds = dl.dataset
                    del ds.tensors
                    del ds
                del dl
        
        # Also delete cached prediction tensors (may be on GPU)
        for attr in ['y_pred_tr', 'y_tr', 'y_pred_v', 'y_v', 'y_pred_te', 'y_te']:
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
        
        # Reset all to None
        self.x_train, self.x_valid, self.x_test, self.y_train, self.y_valid, self.y_test = \
            None, None, None, None, None, None
        self.dl_train, self.dl_valid, self.dl_test = None, None, None
        self.y_pred_tr, self.y_tr, self.y_pred_v, self.y_v, self.y_pred_te, self.y_te = \
            None, None, None, None, None, None
        torch.cuda.empty_cache()
        return

    @staticmethod
    def _metrics_for_multi_dimensions(metric_func, y_pred, y):
        if y.ndim == 2 and y.shape[1] > 1:
            return {'e': metric_func(y_pred[:, 0], y[:, 0]).round(5),
                    'f': metric_func(y_pred[:, 1], y[:, 1]).round(5)}
        else:
            return metric_func(y_pred, y).round(5)

    def calculate_mean_absolute_error(self, y_pred, y):
        return self._metrics_for_multi_dimensions(lambda y1, y2: np.mean(np.abs(y1 - y2)), y_pred, y)

    def calculate_root_mean_squared_error(self, y_pred, y):
        return self._metrics_for_multi_dimensions(lambda y1, y2: np.sqrt(np.mean((y1 - y2) ** 2)), y_pred, y)

    def calculate_correlation_coefficient(self, y_pred, y):
        return self._metrics_for_multi_dimensions(pearson_correlation_coefficient, y_pred, y)

    def calculate_metrics(self, y_pred, y):
        return self.calculate_mean_absolute_error(y_pred, y), self.calculate_root_mean_squared_error(y_pred, y), \
            self.calculate_correlation_coefficient(y_pred, y)


class TrainingWorkflowFromArray(BarelyTrainNN):
    """ Based on CONBQA : CONtinuous Black-box optimization with Quantum Annealing """
    def __init__(self, X, y,
                 epochs, batch_size, lr, lr_s_name, warmup_steps, patience,
                 weight_decay, mt_lambda, alpha, beta, loss_type, ema_momentum, logger,
                 valid_rate, test_rate, x_scaling_method, y_scaling_method,
                 ):
        super().__init__(X=X, y=y, epochs=epochs, batch_size=batch_size, valid_rate=valid_rate, test_rate=test_rate, patience=patience,
                         lr=lr, lr_s_name=lr_s_name, warmup_steps=warmup_steps, weight_decay=weight_decay, mt_lambda=mt_lambda,
                         alpha=alpha, beta=beta, loss_type=loss_type, ema_momentum=ema_momentum, logger=logger)
        
        assert type(self.X) is np.ndarray
        assert self.X.ndim == 2
        self.d = self.X.shape[1]
        assert type(self.y) is np.ndarray

        # transformed data
        self.t_x, self.t_y = None, None

        # scaling settings
        self.x_scaling_method = x_scaling_method
        self.y_scaling_method = y_scaling_method
        self.scaling_subtractor = None
        self.scaling_divisor = None
        
        self.check = not DEBUG

        return

    def _transform_itself(self, dataset, scaling_method):
        if scaling_method == "MinMaxNormalization":
            self.scaling_subtractor = np.min(dataset, axis=0)
            self.scaling_divisor = np.max(dataset, axis=0) - np.min(dataset, axis=0)
        elif scaling_method == "Standardization":
            self.scaling_subtractor = np.mean(dataset, axis=0)
            self.scaling_divisor = np.std(dataset, axis=0)
        elif scaling_method is None or scaling_method == "None":
            self.scaling_subtractor = 0.0
            self.scaling_divisor = 1.0
        else:
            raise Exception("Unexpected scaling method")
        self.scaling_divisor = np.where(self.scaling_divisor > 1e-8, self.scaling_divisor, 1e-8)
        return (dataset - self.scaling_subtractor) / self.scaling_divisor

    def _transform_with_training_set(self, dataset):
        assert self.scaling_subtractor is not None, "Please transform the training set first"
        return (dataset - self.scaling_subtractor) / self.scaling_divisor

    def transform_x(self):
        assert self.x_train is not None, "Please split the dataset first"
        if self.t_x is not None:
            return
        self.t_x = [self._transform_itself(self.x_train, scaling_method=self.x_scaling_method)]
        if not self.no_valid:
            self.t_x.append(self._transform_with_training_set(self.x_valid))
            self.t_x.append(self._transform_with_training_set(self.x_test))
        else:
            self.t_x.append(self._transform_with_training_set(self.x_test))
        return

    def transform_y(self):
        assert self.y_train is not None, "Please split the dataset first"
        if self.t_y is not None:
            return
        self.t_y = [self._transform_itself(self.y_train, scaling_method=self.y_scaling_method)]
        if not self.no_valid:
            self.t_y.append(self._transform_with_training_set(self.y_valid))
            self.t_y.append(self._transform_with_training_set(self.y_test))
        else:
            self.t_y.append(self._transform_with_training_set(self.y_test))
        return

    def transform_y_back(self, y):
        return y * self.scaling_divisor + self.scaling_subtractor
    
    def _evaluate(self, valid_type, dl_to_be_evaluate=None):
        assert valid_type in ['training', 'validation', 'test']
        if dl_to_be_evaluate is None:
            if valid_type == 'training':
                dl_to_be_evaluate = self.dl_train
            elif valid_type == 'validation':
                dl_to_be_evaluate = self.dl_valid
            else:
                dl_to_be_evaluate = self.dl_test
        y_pred, y_real = self.trainer.get_pred_and_real(dl_to_be_evaluate)
        y_pred, y_real = self.transform_y_back(y_pred), self.transform_y_back(y_real)
        log_printer.info('Transforming back to original mean and std')
        mae, rmse, pcc = self.calculate_metrics(y_pred=y_pred, y=y_real)
        if valid_type == 'training':
            if self.check:
                pcc_check = pcc if not isinstance(pcc, dict) else pcc['e']
                if pcc_check <= 0.4:
                    import pickle, bz2
                    from crysim.src import DATASET_ROOT
                    fail_dir = f"{DATASET_ROOT}/{valid_type}-{str(round(pcc_check, 5)).replace('.', 'p')}.pkl.bz2"
                    with bz2.open(fail_dir, "wb") as f:
                        pickle.dump([(d[0].detach().cpu().numpy(), d[1].detach().cpu().numpy()) for d in dl_to_be_evaluate.dataset], f)
                    log_printer.info(f"Failed dataset saved in {fail_dir}")
        if self.logger is not None:
            self.logger.record_valid_metric(mae=mae, rmse=rmse, pcc=pcc, valid_type=valid_type)
        else:
            log_printer.info(f"For the {valid_type} dataset, mae: {mae}, rmse: {rmse}, pcc: {pcc}")
        return y_pred, y_real
    
    def evaluate_all(self):
        if self.no_valid:
            self.y_pred = np.concatenate([self.y_pred_tr, self.y_pred_te])
            self.y_real = np.concatenate([self.y_tr, self.y_te])
        else:
            self.y_pred = np.concatenate([self.y_pred_tr, self.y_pred_v, self.y_pred_te])
            self.y_real = np.concatenate([self.y_tr, self.y_v, self.y_te])
        mae, rmse, pcc = self.calculate_metrics(y_pred=self.y_pred, y=self.y_real)
        if self.logger is not None:
            self.logger.record_pes_metric(mae=mae, rmse=rmse, pcc=pcc)
        else:
            log_printer.info(f"For the whole dataset, mae: {mae}, rmse: {rmse}, pcc: {pcc}")
        return

    def _load_dataset(self):
        self.split()
        self.transform_x()
        self.transform_y()
        self.build_dataset(x_list=self.t_x, y_list=self.t_y)
        return
    
    def set_data(self, x, y):
        self.reset_cash_t()
        self.X, self.y = x, y
        return

    # def add_data(self, x, y):
    #     self.X = np.append(self.X, x.reshape(-1, self.d), axis=0)
    #     self.y = np.append(self.y, y, axis=0)
    #     self.X, self.y, _ = filter_duplicated_embedded_struct(self.X, self.y)
    #     self.reset_cash_t()
    #     if self.logger is not None:
    #         self.logger.save_training_set(X=self.X, y=self.y)
    #     return

    def reset_cash_t(self):
        # Delete transformed data (may contain GPU tensors)
        if self.t_x is not None:
            for tx in self.t_x:
                del tx
        if self.t_y is not None:
            for ty in self.t_y:
                del ty
        self.t_x, self.t_y = None, None
        self.scaling_subtractor, self.scaling_divisor = None, None
        self.reset_cash()
        return

    @property
    def num_var(self):
        return self.d


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    return trainable_parameters, all_param


""" Scientific value function in Matter, 2024, 7, 685-696. """
class SVFLoss(nn.Module):
    def __init__(self, ref_dataset, reduction_func='sum'):
        super().__init__()
        self.ref_x, self.ref_y = ref_dataset
        self.reduction_func = reduction_func
        return

    def forward(self, esti_x, esti_y):
        def batched_euc_dist(m1, m2):
            # m1: (M, D), m2: (N, D)
            m1_sq = (m1**2).sum(dim=1, keepdim=True)        # (M, 1)
            m2_sq = (m2**2).sum(dim=1, keepdim=True).T      # (1, N)
            cross = m1 @ m2.T                               # (M, N)
            dist_sq = m1_sq + m2_sq - 2 * cross
            dist_sq = torch.clamp(dist_sq, min=0)           # numerical stability
            return dist_sq
        gy = torch.sqrt(batched_euc_dist(esti_y, self.ref_y))
        if isinstance(esti_x, (list, tuple)):
            x_size = [p.shape[1] for p in esti_x]
            ref_x = torch.split(self.ref_x, x_size, dim=1)
            hx2 = sum([batched_euc_dist(p, x) for p, x in zip(esti_x, ref_x)])
        else:
            hx2 = batched_euc_dist(esti_x, self.ref_x)
        u = gy * torch.exp(-0.5 * hx2 / torch.clamp(torch.min(hx2, dim=1, keepdim=True).values, 1e-8))
        if self.reduction_func == 'sum':
            u = torch.sum(u, dim=1)
        elif self.reduction_func == 'mean':
            u = torch.mean(u, dim=1)
        else:
            raise NotImplementedError
        return -u
