import math
import platform

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch import optim

from modules.vocab import Vocabulary


class LearningRateScheduler(object):
    """
    Provides inteface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, init_lr, min_lr):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.min_lr = min_lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']
    
    def reset_min_lr(self):
        self.min_lr *= 0.1


class TriStageLRScheduler(LearningRateScheduler):
    """
    Tri-Stage Learning Rate Scheduler
    Implement the learning rate scheduler in "SpecAugment"
    """
    def __init__(self, optimizer, init_lr, peak_lr, final_lr, init_lr_scale, final_lr_scale, warmup_steps, total_steps):
        assert isinstance(warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(total_steps, int), "total_steps should be inteager type"

        super(TriStageLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = int(total_steps >> 1) - warmup_steps
        self.decay_steps = int(total_steps >> 1)

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0
        # Keeping track of loss
        self.prev_loss = float('inf')

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        offset = self.warmup_steps

        if self.update_step < offset + self.hold_steps:
            return 1, self.update_step - offset

        offset += self.hold_steps

        if self.update_step <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_step - offset

        offset += self.decay_steps

        return 3, self.update_step - offset

    def step(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_step += 1

        return self.lr

class LossAwareLRScheduler(LearningRateScheduler):
    """
    Starts with init_lr, increases linearly to peak_lr over warmup_steps, and 
    then adjusts the learning rate based on loss changes.
    """
    def __init__(self, optimizer, init_lr, peak_lr, min_lr, warmup_steps, decay_steps, reduction_factor=0.9, patience=2):
        super(LossAwareLRScheduler, self).__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.reduction_factor = reduction_factor
        self.patience = patience
        self.patience_counter = 0
        self.prev_loss = float('inf')
        self.decay_steps = decay_steps
        self.min_lr = min_lr # (self.init_lr)**2 / self.peak_lr
    def step(self, current_loss):
        # Linear warmup phase
        if self.current_step < self.warmup_steps:
            lr = self.init_lr + (self.peak_lr - self.init_lr) * (self.current_step / self.warmup_steps)
            lr = min(self.peak_lr, lr)
            self.set_lr(self.optimizer, lr)
        # Adjusting phase based on loss
        else:
            step = self.current_step - self.warmup_steps
            if step % self.decay_steps == 0:
                self.reset_min_lr()
            # If current loss is greater than the previous loss
            if current_loss > self.prev_loss:
                self.patience_counter += 1
                # If patience is exhausted, reduce the learning rate
                if self.patience_counter >= self.patience:
                    self.patience_counter = 0
                    new_lr = self.get_lr() * self.reduction_factor
                    new_lr = max(new_lr, self.min_lr)
                    self.set_lr(self.optimizer, new_lr)
                    
            else:
                # Reset the patience counter if loss decreases or stays the same
                new_lr = self.get_lr() / self.reduction_factor
                new_lr = min(new_lr, self.peak_lr)
                self.set_lr(self.optimizer, new_lr)
                self.patience_counter = 0
        
        # Update the previous loss and current_step for the next iteration
        self.prev_loss = current_loss
        self.current_step += 1
        


class Optimizer(object):
    """
    This is wrapper classs of torch.optim.Optimizer.
    This class provides functionalities for learning rate scheduling and gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): optimizer object, the parameters to be optimized
            should be given when instantiating the object, e.g. torch.optim.Adam, torch.optim.SGD
        scheduler (kospeech.optim.lr_scheduler, optional): learning rate scheduler
        scheduler_period (int, optional): timestep with learning rate scheduler
        max_grad_norm (int, optional): value used for gradient norm clipping
    """
    def __init__(self, optim, scheduler=None, scheduler_period=None, max_grad_norm=0):
        self.optimizer = optim
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.max_grad_norm = max_grad_norm
        self.count = 0

    def step(self, model, current_loss):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.scheduler is not None:
            self.update(current_loss)
            self.count += 1

            if self.scheduler_period == self.count:
                self.scheduler = None
                self.scheduler_period = 0
                self.count = 0

    def set_scheduler(self, scheduler, scheduler_period):
        self.scheduler = scheduler
        self.scheduler_period = scheduler_period
        self.count = 0

    def update(self, current_loss):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass
        else:
            self.scheduler.step(current_loss)
    
    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr


def get_lr_scheduler(config, optimizer, epoch_time_step) -> LearningRateScheduler:
    if config.lr_scheduler == 'tri_stage_scheduler':        
        lr_scheduler = TriStageLRScheduler(
            optimizer=optimizer,
            init_lr=config.init_lr,
        )
    else:
        lr_scheduler = LossAwareLRScheduler(
            optimizer=optimizer,
            init_lr=config.init_lr,
            peak_lr=config.peak_lr,
            min_lr = config.min_lr,
            #min_lr=config.final_lr,
            #init_lr_scale=config.init_lr_scale,
            #final_lr_scale=config.final_lr_scale,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            patience=config.lr_patience,
            #total_steps=int(config.num_epochs * epoch_time_step),
        )

    return lr_scheduler


def get_optimizer(model: nn.Module, config):
    supported_optimizer = {
        'adam': optim.Adam,
    }

    return supported_optimizer[config.optimizer](
        model.module.parameters(),
        lr=config.init_lr,
        weight_decay=config.weight_decay,
    )


def get_criterion(config, vocab: Vocabulary) -> nn.Module:

    criterion = nn.CTCLoss(blank=vocab.blank_id, reduction=config.reduction, zero_infinity=True)

    return criterion

def revise(sentence: str):
    assert type(sentence) == str, "Input is not a string"
    words = sentence[0].split()
    result = []
    for word in words:
        tmp = ''    
        for t in word:
            if not tmp:
                tmp += t
            elif tmp[-1]!= t:
                tmp += t
        # if tmp == '스로':
        #     tmp = '스스로'
        result.append(tmp)
    return ' '.join(result)
