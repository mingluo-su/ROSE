import torch
import torch.nn as nn

from .prune_zoo.wanda import Wanda
from .prune_zoo.dsnot import DSnoT
from .prune_zoo.sparsegpt import SparseGPT
from .prune_zoo.rose import ROSE
from .utils import find_layers
from .data import get_loaders


def prepare_calibration_input(args, model, dataloader, device):
    """
    Collect calibration inputs for layer-wise pruning.

    Returns:
        inps: calibration inputs
        outs: placeholder outputs
        attention_mask
        position_embeddings
    """

    use_cache = getattr(model.config, "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = False

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError("Model must contain model.model.layers")

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
        model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(device)

    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "dim", None)
        if hidden_size is None:
            raise ValueError("Cannot find hidden_size or dim in model config")

    inps = torch.zeros(
        (args.nsamples, model.seqlen, hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False

    cache = {"i": 0, "attention_mask": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] < args.nsamples:
                inps[cache["i"]] = inp.detach()

            cache["i"] += 1
            if "attention_mask" in kwargs:
                cache["attention_mask"] = kwargs["attention_mask"]
            if "position_embeddings" in kwargs:
                cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError("Catcher stop forward")

    original_first_layer = layers[0]
    layers[0] = Catcher(layers[0])

    samples_collected = 0
    for batch in dataloader:
        if samples_collected >= args.nsamples:
            break
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
        samples_collected = min(cache["i"], args.nsamples)

    layers[0] = original_first_layer
    layers[0] = layers[0].to("cpu")
    model.model.embed_tokens = model.model.embed_tokens.to("cpu")
    model.model.norm = model.model.norm.to("cpu")

    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to("cpu")

    outs = torch.zeros_like(inps)
    if use_cache is not None:
        model.config.use_cache = use_cache

    torch.cuda.empty_cache()

    return inps, outs, cache["attention_mask"], cache["position_embeddings"]


@torch.no_grad()
def prune_model(args, model, tokenizer, device=torch.device("cuda"), prune_n=0, prune_m=0):
    """
    Layer-wise pruning pipeline.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if not hasattr(model.model, "layers"):
        raise ValueError("Model must contain model.model.layers")

    layers = model.model.layers

    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
    )
    inps, outs, attention_mask, position_embeddings = prepare_calibration_input(
        args, model, dataloader, device
    )

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    if isinstance(position_embeddings, tuple):
        position_embeddings = tuple(e.to(device) for e in position_embeddings)

    if args.prune_method == "magnitude":
        prune_fn = prune_magnitude
    elif args.prune_method == "wanda":
        prune_fn = prune_wanda
    elif args.prune_method in ["sparsegpt", "rose"]:
        prune_fn = prune_sparsegpt
    elif args.prune_method == "dsnot":
        prune_fn = prune_dsnot
    else:
        raise ValueError(f"Unsupported prune_method: {args.prune_method}")

    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer)

        wrapped_layers = {}

        for name in subset:
            if args.prune_method == "magnitude":
                wrapped_layers[name] = None
            elif args.prune_method == "wanda":
                wrapped_layers[name] = Wanda(subset[name])
            elif args.prune_method == "sparsegpt":
                wrapped_layers[name] = SparseGPT(subset[name])
            elif args.prune_method == "dsnot":
                wrapped_layers[name] = DSnoT(subset[name], layer_name=name)
            elif args.prune_method == "rose":
                wrapped_layers[name] = ROSE(subset[name])
            else:
                raise ValueError("Invalid prune_method during wrapping")

        handles = []
        if args.prune_method in ["wanda", "sparsegpt", "rose", "dsnot"]:
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(device),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        for h in handles:
            h.remove()

        for name in subset:
            prune_fn(
                subset[name],
                wrapped_layers[name],
                args.sparsity_ratio,
                prune_n,
                prune_m,
            )
            print(f"Pruning layer {i} - {name}")

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(device),
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

        inps, outs = outs, inps
        layers[i] = layer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# =========================================================
#  Pruning Methods
# =========================================================

def prune_magnitude(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    magnitude pruning.
    """
    if layer is None:
        raise ValueError("Layer cannot be None")

    layer = layer.to("cuda")
    W = layer.weight.data
    W_metric = torch.abs(W)

    if prune_n != 0:
        W_mask = torch.zeros_like(W_metric) == 1
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                idx = torch.topk(tmp, prune_n, dim=1, largest=True)[1]
                W_mask.scatter_(1, ii + idx, True)
    else:
        thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * sparsity_ratio)]
        W_mask = W_metric > thresh

    layer.weight.data[W_mask] = 0


def prune_wanda(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    Wanda pruning.
    """
    if wrapped_layer is None:
        raise ValueError("wrapped_layer cannot be None for Wanda")

    W_metric = torch.abs(layer.weight.data) * torch.sqrt(wrapped_layer.scaler_row.reshape((1, -1)))
    W_mask = torch.zeros_like(W_metric) == 1

    if prune_n != 0:
        for ii in range(W_metric.shape[1]):
            if ii % prune_m == 0:
                tmp = W_metric[:, ii:(ii + prune_m)].float()
                idx = torch.topk(tmp, prune_n, dim=1, largest=False)[1]
                W_mask.scatter_(1, ii + idx, True)
    else:
        sort_res = torch.sort(W_metric, dim=-1, stable=True, descending=True)
        indices = sort_res[1][:, :int(W_metric.shape[1] * (1 - sparsity_ratio))]
        W_mask.scatter_(1, indices, True)

    layer.weight.data[W_mask] = 0
    wrapped_layer.free()


def prune_sparsegpt(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    SparseGPT / ROSE pruning.
    """
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for SparseGPT/ROSE")

    wrapped_layer.fasterprune(
        sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        percdamp=0.01,
        blocksize=128,
    )
    wrapped_layer.free()


def prune_dsnot(layer, wrapped_layer, sparsity_ratio, prune_n=0, prune_m=0):
    """
    DSnoT pruning.
    """
    if wrapped_layer is None:
        raise ValueError("wrapped_layer required for DSnoT")

    wrapped_layer.fasterprune(
        sparsity=sparsity_ratio,
        prune_n=prune_n,
        prune_m=prune_m,
        max_cycle_time=50,
        update_threshold=0.1,
        pow_of_var_regrowing=1.0,
        pow_of_var_pruning=1.0,
        without_DSnoT=False,
        skip_layer="mlp",
        skip_sub_layer="no_skip",
        without_same_sign=True,
    )
    wrapped_layer.free()