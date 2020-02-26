import math
from typing import List

import pretrainedmodels
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, init
from torch.nn.parameter import Parameter


IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]


def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out


class LazyLoadModule(nn.Module):
    """Lazy buffer/parameter loading using load_state_dict_pre_hook

    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and
    save buffer with `register_buffer`/`register_parameter`
    method, which can be outside of __init__ method.
    Then this module can load any shape of Tensor during de-serializing.

    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.
    """
    _lazy_buffer_keys: List[str] = []     # It needs to be override to register lazy buffer
    _lazy_parameter_keys: List[str] = []  # It needs to be override to register lazy parameter

    def __init__(self):
        super(LazyLoadModule, self).__init__()
        for k in self._lazy_buffer_keys:
            self.register_buffer(k, torch.tensor([]))
        for k in self._lazy_parameter_keys:
            self.register_parameter(k, None)
        self._register_load_state_dict_pre_hook(self._hook)

    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])

        for key in self._lazy_parameter_keys:
            self.register_parameter(key, Parameter(state_dict[prefix + key]))


class LazyLinear(LazyLoadModule):
    """Linear module with lazy input inference

    `in_features` can be `None`, and it is determined at the first time of forward step dynamically.
    """

    __constants__ = ['bias', 'in_features', 'out_features']
    _lazy_parameter_keys = ['weight']

    def __init__(self, in_features, out_features, bias=True):
        super(LazyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if in_features is not None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is None:
            self.in_features = input.shape[-1]
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.reset_parameters()

            # Need to send lazy defined parameter to device...
            self.to(input.device)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        if in_features is None:
            self.linear = LazyLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h


class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=1, out_dim=10, use_bn=True,
                 pretrained='imagenet',
                 n_grapheme=168, n_vowel=11, n_consonant=7):
        super(PretrainedCNN, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        activation = F.leaky_relu
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = n_grapheme + n_vowel + n_consonant

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        
        pred = h

        if isinstance(pred, tuple):
            assert len(pred) == 3
            preds = pred
        else:
            assert pred.shape[1] == self.n_total_class
            preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)

        logit_grapheme_root = preds[0]
        logit_vowel_diacritic = preds[1]
        logit_consonant_diacritic = preds[2]
        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


class Net_1(nn.Module):
    def __init__(self, num_class=(168, 11, 7),
                pretrained='imagenet',
                model_name='se_resnext101_32x4d'):
        super(Net_1, self).__init__()
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        self.base_model.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.base_model.last_linear = nn.ModuleList(
        #     [nn.Linear(self.base_model.last_linear.in_features, c) for c in num_class]
        # )
        self.base_model.last_linear = nn.Linear(self.base_model.last_linear.in_features, 186)

    def forward(self, x):
        h = self.base_model(x.repeat(1, 3, 1, 1))
        logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = h[:,: 168], h[:,168: 168+11], h[:,168+11:]
        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


class Net(nn.Module):
    def __init__(self, num_class=(168, 11, 7),
                 pretrained='imagenet',
                 model_name='se_resnext101_32x4d'):
        super(Net, self).__init__()
        e = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        # in_features = e.in_features
        self.block0 = e.layer0
        self.block1 = e.layer1
        self.block2 = e.layer2
        self.block3 = e.layer3
        self.block4 = e.layer4
        e = None  #dropped

        self.logit = nn.ModuleList(
            [nn.Linear(2048, c) for c in num_class]
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.block0(x.repeat(1, 3, 1, 1))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.2, self.training)

        logit = [l(x) for l in self.logit]
        logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic = logit[0], logit[1], logit[2]
        return logit_grapheme_root, logit_vowel_diacritic, logit_consonant_diacritic


def get_model(config):
    if config.model.arch == "net0":
        model =  PretrainedCNN(in_channels=1,
                            out_dim=186,
                            model_name=config.model.version,
                            pretrained=config.model.pretrained
                            )
    elif config.model.arch == "net1":
        model = Net(model_name=config.model.version, pretrained=config.model.pretrained)
    elif(config.model.arch) == "net2":
        model = Net_1(model_name=config.model.version, pretrained=config.model.pretrained)
    return model
