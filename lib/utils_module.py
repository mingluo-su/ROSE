import os
import psutil
import torch
import torch.nn as nn
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

np.random.seed(0)
torch.manual_seed(0)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):

    count = 0
    total_params = 0
    sub_count = 0
    sub_params = 0

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "OPT" in model.__class__.__name__:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    return float(count) / total_params


