import math

import torch
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, MultiStepLR,
                                      _LRScheduler, ReduceLROnPlateau)

from .scheduler import Scheduler


class PlateauLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,
                 factor=0.1,
                 patience=10,
                 verbose=False,
                 threshold=1e-4,
                 cooldown_epochs=0,
                 warmup_updates=0,
                 warmup_lr_init=0,
                 lr_min=0,
                 ):
        super().__init__(optimizer, 'lr', initialize=False)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer,
            patience=patience,
            factor=factor,
            verbose=verbose,
            threshold=threshold,
            cooldown=cooldown_epochs,
            min_lr=lr_min
        )

        self.warmup_updates = warmup_updates
        self.warmup_lr_init = warmup_lr_init

        if self.warmup_updates:
            self.warmup_active = warmup_updates > 0  # this state updates with num_updates
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_updates for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def state_dict(self):
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    # override the base class step fn completely
    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None and not self.warmup_active:
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.last_epoch = epoch

    def get_update_values(self, num_updates: int):
        if num_updates < self.warmup_updates:
            lrs = [self.warmup_lr_init + num_updates * s for s in self.warmup_steps]
        else:
            self.warmup_active = False  # warmup cancelled by first update past warmup_update count
            lrs = None  # no change on update after warmup stage
        return lrs


class HalfCosineAnnealingLR(_LRScheduler):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.T_max = T_max
        super(HalfCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % (2 * self.T_max) < self.T_max:
            cos_unit = 0.5 * \
                (math.cos(math.pi * self.last_epoch / self.T_max) - 1)
        else:
            cos_unit = 0.5 * \
                (math.cos(math.pi * (self.last_epoch / self.T_max - 1)) - 1)

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * 1.0e-4
            range = math.log10(base_lr - math.log10(min_lr))
            lrs.append(10 ** (math.log10(base_lr) + range * cos_unit))
        return lrs


def get_scheduler(optimizer, config):
    if config.scheduler.name == 'plateau':
        # scheduler = PlateauLRScheduler(optimizer, factor=0.7, patience=3, lr_min=1e-10)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=7, min_lr=1e-10, verbose=True)
    elif config.scheduler.name == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-2, total_steps=None, epochs=config.train.num_epochs, steps_per_epoch=5021, pct_start=0.0, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=100.0)
    elif config.scheduler.name == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, 5, eta_min=1e-6, last_epoch=-1)
    elif config.scheduler.name == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler.params.t_max, eta_min=1e-6,
                                                last_epoch=-1)
    elif config.scheduler.name == 'half_cosine':
        scheduler = HalfCosineAnnealingLR(
            optimizer, config.scheduler.params.t_max, last_epoch=-1)
    elif config.scheduler.name == 'multi_step':
        scheduler = MultiStepLR(optimizer, milestones=[35, 70, 100, 110,  120, 130, 140], gamma=0.6)
    else:
        scheduler = None
    return scheduler
