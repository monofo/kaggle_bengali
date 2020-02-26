'''
https://github.com/rwightman/pytorch-image-models
'''
import torch
from torch import optim as optim
# from timm.optim import Nadam, RMSpropTF, AdamW, RAdam, NovoGrad, NvNovoGrad, Lookahead
from optimizers import AdamW, RAdam, Lookahead
try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def get_optimizer(config, model, filter_bias_and_bn=True):
    opt_lower = config.optimizer.name.lower()
    weight_decay = config.optimizer.params.weight_decay
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        weight_decay /= config.optimizer.params.lr
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_look_ahed = config.optimizer.lookahead.apply
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters, lr=config.optimizer.params.lr, momentum=config.optimizer.params.momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            parameters, lr=config.optimizer.params.lr, weight_decay=weight_decay, eps=config.optimizer.params.opt_eps)
    elif opt_lower == 'adamw':
        optimizer = AdamW(
            parameters, lr=config.optimizer.params.lr, weight_decay=weight_decay, eps=config.optimizer.params.opt_eps)
    elif opt_lower == 'nadam':
        optimizer = Nadam(
            parameters, lr=config.optimizer.params.lr, weight_decay=weight_decay, eps=config.optimizer.params.opt_eps)
    elif opt_lower == 'radam':
        optimizer = RAdam(
            parameters, lr=config.optimizer.params.lr, weight_decay=weight_decay, eps=config.optimizer.params.opt_eps)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if opt_look_ahed:
        optimizer = Lookahead(optimizer)

    return optimizer