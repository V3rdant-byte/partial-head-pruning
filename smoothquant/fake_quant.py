import torch
from torch import nn
from functools import partial


def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_per_head_absmax(w, less_important_indices, n_bits=8):
    w_shape = w.data.shape
    num_heads = 32
    head_dim =  int(w.shape[1] / num_heads)
    w.data = w.data.view(w_shape[0], num_heads, head_dim).transpose(0, 1).contiguous()
    w.data = w.data.view(num_heads, -1)
    q_max = 2 ** (n_bits - 1) - 1
    for index in less_important_indices:
        scales = w.data[index].abs().max(dim=-1, keepdim=True)[0]
        scales.clamp_(min=1e-5).div_(q_max)
        w.data[index].div_(scales).round_().mul_(scales)
    w.data = w.data.view(num_heads, w_shape[0], head_dim).transpose(0, 1).contiguous()
    w.data = w.data.view(w_shape[0], w_shape[1])
    return w

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]    # max value for this token only
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2**(n_bits-1)-1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_head_absmax(t, indices_to_quantize, n_bits=8):
    t_shape = t.shape     # 1 * 512 * 2048
    num_heads = 32
    head_dim = int(t_shape[-1] / num_heads)
    act = t.view(t_shape[-2], num_heads, head_dim).transpose(0, 1).contiguous()
    act = act.view(num_heads, -1) # 32 * 32768
    q_max = 2 ** (n_bits - 1) - 1
    for index in indices_to_quantize:
        scales = act[index].abs().max(dim=-1, keepdim=True)[0]  # 32 * 1
        scales.clamp_(min=1e-5).div_(q_max)
        act[index].div_(scales).round_().mul_(scales)
    #t.div_(scales).round_().mul_(scales)
    act = act.view(num_heads, t_shape[-2], head_dim).transpose(0, 1).contiguous()
    act = act.view(t_shape)
    return act

class W8A8Linear(nn.Module):
    def __init__(self, in_features, out_features, indices_to_quantize, bias=True, act_quant='per_token', quantize_output=True, output_attentions=False, is_attention=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.indices_to_quantize = indices_to_quantize
        self.quantize_output = quantize_output
        self.is_attention = is_attention
        self.is_per_head = (act_quant == 'per_head')

        self.register_buffer('weight', torch.randn(self.out_features,
                                                   self.in_features, dtype=torch.float16, requires_grad=False))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (1, self.out_features), dtype=torch.float16, requires_grad=False))
        else:
            self.register_buffer('bias', None)

        if act_quant == 'per_token':
            self.act_quant_name = 'per_token'
            self.act_quant = partial(
                quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == 'per_tensor':
            self.act_quant_name = 'per_tensor'
            self.act_quant = partial(
                quantize_activation_per_tensor_absmax, n_bits=8)
        elif act_quant == 'per_head':
            self.act_quant_name = 'per_head'
            self.act_quant = partial(
                quantize_activation_per_head_absmax, n_bits=8)
        else:
            raise ValueError(f'Invalid act_quant: {act_quant}')

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = 'None'
            self.output_quant = lambda x: x

        if output_attentions:
            self.output_attentions = output_attentions

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.is_per_head:
            #if self.is_attention:
            q_x = self.act_quant(x, self.indices_to_quantize)
            #else:
            #    q_x = self.act_quant(x)
            y = torch.functional.F.linear(q_x, self.weight, self.bias)
            if self.quantize_output:
                q_y = self.output_quant(y, self.indices_to_quantize)
            else:
                q_y = self.output_quant(y)
        else:
            q_x = self.act_quant(x)
            y = torch.functional.F.linear(q_x, self.weight, self.bias)
            q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(module, less_important_indices, weight_quant='per_channel', act_quant='per_token', quantize_output=False, output_attentions=False, is_attention=False):
        # assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features, module.out_features, less_important_indices, module.bias is not None, act_quant=act_quant, quantize_output=quantize_output, output_attentions=output_attentions, is_attention=is_attention)
        if weight_quant == 'per_channel':
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8)  # use 8-bit integer for weight
        elif weight_quant == 'per_tensor':
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8)
        elif weight_quant == 'per_head':
            new_module.weight = quantize_weight_per_head_absmax(
                module.weight, less_important_indices, n_bits=8)
        else:
            raise ValueError(f'Invalid weight_quant: {weight_quant}')
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f'W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})'
