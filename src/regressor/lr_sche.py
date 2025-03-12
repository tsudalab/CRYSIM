import numpy as np
from torch.optim import lr_scheduler


class WarmupSche:
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        return

    def __call__(self):
        if self.warmup_steps > 0:
            self.wus = lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=1 / self.warmup_steps,
                                             end_factor=1, total_iters=self.warmup_steps)
            return self.wus
        else:
            return None

    def get_lr(self):
        return self.wus.get_last_lr()


class LRSche:
    def __init__(self, optimizer, epoch, start_lr=None, end_lr=None):
        self.optimizer = optimizer
        self.epoch = epoch
        self.start_lr = start_lr if start_lr is not None else 1e-4
        self.end_lr = end_lr if end_lr is not None else self.start_lr * 1e-4
        self.lrs = None
        return

    def _plateau(self):
        lrs = lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min',
                                             factor=0.9, patience=10, min_lr=self.end_lr)
        return lrs

    def _linear(self):
        lrs = lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=1,
                                    end_factor=self.end_lr / self.start_lr,
                                    total_iters=int(self.epoch * 0.8))
        return lrs

    def _exponential(self):
        gamma = 0.99
        milestone = int(np.log(self.end_lr / self.start_lr) / np.log(gamma))
        factor = self.end_lr / self.start_lr
        lrs = lr_scheduler.SequentialLR(optimizer=self.optimizer,
                                        schedulers=[
                                            lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                       gamma=gamma),
                                            lr_scheduler.ConstantLR(optimizer=self.optimizer, factor=factor,
                                                                    total_iters=self.epoch)
                                        ], milestones=[milestone])
        return lrs

    def _multistep(self):
        gamma = 0.1
        num_stone = int(np.log(self.end_lr / self.start_lr) / np.log(gamma))
        stone_step = int(self.epoch / (num_stone + 1))
        milestone = [stone_step * i for i in range(1, num_stone + 1)]
        lrs = lr_scheduler.MultiStepLR(optimizer=self.optimizer, gamma=gamma, milestones=milestone)
        return lrs

    def _sequence(self):
        milestone = int(self.epoch * 0.4)
        # linear_end_lr = self.start_lr * float(milestone / self.epoch)
        lrs = lr_scheduler.SequentialLR(optimizer=self.optimizer,
                                        schedulers=[
                                            lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                  start_factor=1,
                                                                  end_factor=self.end_lr / self.start_lr,
                                                                  total_iters=int(milestone * 0.8)),
                                            lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                       gamma=0.993)
                                        ], milestones=[milestone])
        return lrs

    def __call__(self, name):
        if name == "ReduceLROnPlateau":
            self.lrs = self._plateau()
        elif name == "LinearLR":
            self.lrs = self._linear()
        elif name == "ExponentialLR":
            self.lrs = self._exponential()
        elif name == "MultiStepLR":
            self.lrs = self._multistep()
        elif name == "SequentialLR":
            self.lrs = self._sequence()
        elif name == "None":
            pass
        return self.lrs

    def get_lr(self):
        return self.lrs.get_last_lr()
