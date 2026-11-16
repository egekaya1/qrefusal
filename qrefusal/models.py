"""Load full-precision and quantized models, and swap weights between them."""

from __future__ import annotations

import copy
from typing import Iterable

import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load as _load

from .hooks import _inner

# Component groups used for weight restoration (RQ2).
GROUPS = {
    "attn_qk": ("self_attn.q_proj", "self_attn.k_proj"),
    "attn_vo": ("self_attn.v_proj", "self_attn.o_proj"),
    "mlp": ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
}


def load(path: str):
    """Return (model, tokenizer) via mlx-lm. Works for bf16 and quantized dirs."""
    model, tok = _load(path)
    model.eval()
    return model, tok


def _get(module, dotted: str):
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def _set(module, dotted: str, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], value)


def _as_linear(src) -> nn.Linear:
    """Copy a full-precision nn.Linear taken from the bf16 model."""
    if isinstance(src, nn.QuantizedLinear):
        raise TypeError("restoration source must be full precision")
    out_dim, in_dim = src.weight.shape
    lin = nn.Linear(in_dim, out_dim, bias="bias" in src)
    lin.weight = src.weight
    if "bias" in src:
        lin.bias = src.bias
    return lin


def _shallow_block(block):
    """Copy a decoder block and its attn/mlp submodules without copying weights."""
    b = copy.copy(block)
    b.self_attn = copy.copy(block.self_attn)
    b.mlp = copy.copy(block.mlp)
    return b


def restore(qmodel, fmodel, layers: Iterable[int], groups: Iterable[str], io: bool = False):
    """Return a copy of qmodel with (layer, group) weights taken from fmodel.

    groups are keys of GROUPS. If io is True, the embedding and LM head are
    restored too. Untouched modules are shared with qmodel, so the copy costs
    only the memory of the restored weights.
    """
    groups = list(groups)
    q = copy.copy(qmodel)
    qi, fi = _inner(q), _inner(fmodel)
    inner = copy.copy(qi)
    new_layers = list(qi.layers)
    for L in layers:
        block = _shallow_block(qi.layers[L])
        for g in groups:
            for name in GROUPS[g]:
                _set(block, name, _as_linear(_get(fi.layers[L], name)))
        new_layers[L] = block
    inner.layers = new_layers
    if io:
        inner.embed_tokens = fi.embed_tokens
        if getattr(fmodel, "lm_head", None) is not None:
            q.lm_head = fmodel.lm_head
    if hasattr(q, "model"):
        q.model = inner
        return q
    return inner


def weight_bytes(model) -> int:
    """Bytes of all parameters (quantized weights, scales and biases included)."""
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))
