"""Layer-by-layer forward pass with capture, patching and vector addition.

MLX has no hook API, and monkey-patching a module instance's __call__ is
silently ignored (ml-explore/mlx#3387). Instead of wrapping modules, this file
re-implements the decoder loop using the model's own submodules. It supports
architectures whose blocks are:

    h = x + self_attn(input_layernorm(x))
    out = h + mlp(post_attention_layernorm(h))

which covers Qwen2/Qwen2.5, Qwen3 and Llama 3 in mlx-lm. Gemma 3 has extra
norms and is not supported here yet.

Sites, per layer index L:
    "resid_pre"  residual stream entering layer L
    "attn"       attention output added to the residual in layer L
    "mlp"        MLP output added to the residual in layer L
    "resid_post" residual stream leaving layer L
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

SITES = ("resid_pre", "attn", "mlp", "resid_post")


@dataclass
class HookSpec:
    """What to do during one forward pass.

    capture: set of (layer, site) whose activations are returned.
    patch:   {(layer, site): (positions, values)}. The activation at those
             positions is replaced by `values`, shape [len(positions), d] or
             [batch, len(positions), d]. positions=None replaces all positions,
             and then values must have the full activation shape.
    add:     {(layer, site): (positions, vector)}. Adds `vector` (shape [d])
             at those positions (None means all positions).
    fn:      {(layer, site): callable(act) -> act}, for anything else
             (for example directional ablation).
    """

    capture: set = field(default_factory=set)
    patch: dict = field(default_factory=dict)
    add: dict = field(default_factory=dict)
    fn: dict = field(default_factory=dict)


def _apply(act: mx.array, key, spec: HookSpec, store: dict) -> mx.array:
    if key in spec.patch:
        pos, vals = spec.patch[key]
        if pos is None:
            act = vals.astype(act.dtype)
        else:
            idx = mx.array(pos)
            v = vals if vals.ndim == 3 else mx.broadcast_to(vals, (act.shape[0],) + vals.shape)
            act = _set_positions(act, idx, v.astype(act.dtype))
    if key in spec.add:
        pos, vec = spec.add[key]
        vec = vec.astype(act.dtype)
        if pos is None:
            act = act + vec
        else:
            delta = mx.zeros_like(act)
            delta = _set_positions(delta, mx.array(pos), mx.broadcast_to(vec, (act.shape[0], len(pos), act.shape[-1])))
            act = act + delta
    if key in spec.fn:
        act = spec.fn[key](act)
    if key in spec.capture:
        store[key] = act
    return act


def _set_positions(act: mx.array, idx: mx.array, vals: mx.array) -> mx.array:
    # act: [B, T, D]; idx: [k]; vals: [B, k, D]
    T = act.shape[1]
    onehot = (mx.arange(T)[:, None] == idx[None, :]).astype(act.dtype)  # [T, k]
    keep = 1.0 - onehot.sum(axis=1)  # [T]
    scattered = mx.einsum("tk,bkd->btd", onehot, vals)
    return act * keep[None, :, None] + scattered


def _inner(model):
    """Return the module that owns embed_tokens / layers / norm."""
    return model.model if hasattr(model, "model") else model


def _unembed(model, h: mx.array) -> mx.array:
    inner = _inner(model)
    if getattr(model, "lm_head", None) is not None:
        return model.lm_head(h)
    return inner.embed_tokens.as_linear(h)


def forward(model, tokens: mx.array, spec: Optional[HookSpec] = None):
    """Run the model on `tokens` ([B, T] int array) with no KV cache.

    Returns (logits [B, T, V], captured {(layer, site): array}).
    """
    spec = spec or HookSpec()
    inner = _inner(model)
    store: dict = {}
    h = inner.embed_tokens(tokens)
    mask = create_attention_mask(h, None)
    for L, layer in enumerate(inner.layers):
        h = _apply(h, (L, "resid_pre"), spec, store)
        a = layer.self_attn(layer.input_layernorm(h), mask, None)
        a = _apply(a, (L, "attn"), spec, store)
        h = h + a
        m = layer.mlp(layer.post_attention_layernorm(h))
        m = _apply(m, (L, "mlp"), spec, store)
        h = h + m
        h = _apply(h, (L, "resid_post"), spec, store)
    logits = _unembed(model, inner.norm(h))
    return logits, store


def n_layers(model) -> int:
    return len(_inner(model).layers)


def directional_ablation(direction: mx.array) -> Callable[[mx.array], mx.array]:
    """Return fn that removes the component of act along unit `direction`."""
    d = direction / mx.linalg.norm(direction)

    def f(act):
        proj = (act @ d.astype(act.dtype))[..., None]
        return act - proj * d.astype(act.dtype)

    return f
