# Lookahead implementation from https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/lookahead.py

__all__ = ["Lookahead", "LookaheadAdam"]

""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
"""
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def LookaheadAdam(params, alpha=0.5, k=6, *args, **kwargs):
     adam = Adam(params, *args, **kwargs)
     return Lookahead(adam, alpha, k)

# from collections import defaultdict

# import torch
# from torch.optim.optimizer import Optimizer


# class Lookahead(Optimizer):
#     r"""PyTorch implementation of the lookahead wrapper.
#     Lookahead Optimizer: https://arxiv.org/abs/1907.08610
#     """

#     def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
#         """optimizer: inner optimizer
#         la_steps (int): number of lookahead steps
#         la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
#         pullback_momentum (str): change to inner optimizer momentum on interpolation update
#         """
#         self.optimizer = optimizer
#         self._la_step = 0  # counter for inner optimizer
#         self.la_alpha = la_alpha
#         self._total_la_steps = la_steps
#         pullback_momentum = pullback_momentum.lower()
#         assert pullback_momentum in ["reset", "pullback", "none"]
#         self.pullback_momentum = pullback_momentum

#         self.state = defaultdict(dict)

#         # Cache the current optimizer parameters
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 param_state['cached_params'] = torch.zeros_like(p.data)
#                 param_state['cached_params'].copy_(p.data)
#                 if self.pullback_momentum == "pullback":
#                     param_state['cached_mom'] = torch.zeros_like(p.data)

#     # def __setstate__(self, state):
#     #     super(Lookahead, self).__setstate__(state)

#     def __getstate__(self):
#         return {
#             'state': self.state,
#             'optimizer': self.optimizer,
#             'la_alpha': self.la_alpha,
#             '_la_step': self._la_step,
#             '_total_la_steps': self._total_la_steps,
#             'pullback_momentum': self.pullback_momentum
#         }

#     def zero_grad(self):
#         self.optimizer.zero_grad()

#     def get_la_step(self):
#         return self._la_step

#     def state_dict(self):
#         return self.optimizer.state_dict()

#     def load_state_dict(self, state_dict):
#         self.optimizer.load_state_dict(state_dict)

#     def _backup_and_load_cache(self):
#         """Useful for performing evaluation on the slow weights (which typically generalize better)
#         """
#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 param_state['backup_params'] = torch.zeros_like(p.data)
#                 param_state['backup_params'].copy_(p.data)
#                 p.data.copy_(param_state['cached_params'])

#     def _clear_and_load_backup(self):
#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 p.data.copy_(param_state['backup_params'])
#                 del param_state['backup_params']

#     @property
#     def param_groups(self):
#         return self.optimizer.param_groups

#     def step(self, closure=None):
#         """Performs a single Lookahead optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = self.optimizer.step(closure)
#         self._la_step += 1

#         if self._la_step >= self._total_la_steps:
#             self._la_step = 0
#             # Lookahead and cache the current optimizer parameters
#             for group in self.optimizer.param_groups:
#                 for p in group['params']:
#                     param_state = self.state[p]
#                     p.data.mul_(self.la_alpha).add_(1.0 - self.la_alpha, param_state['cached_params'])  # crucial line
#                     param_state['cached_params'].copy_(p.data)
#                     if self.pullback_momentum == "pullback":
#                         internal_momentum = self.optimizer.state[p]["momentum_buffer"]
#                         self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
#                             1.0 - self.la_alpha, param_state["cached_mom"])
#                         param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
#                     elif self.pullback_momentum == "reset":
#                         self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

#         return loss