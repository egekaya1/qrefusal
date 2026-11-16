"""Refusal directions and their geometry across precisions."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .data import chat_ids, post_instruction_positions
from .hooks import HookSpec, forward, n_layers


def acts_at(model, tok, instructions, site: str = "resid_pre", where: str = "last"):
    """Activations per layer at the last token ('last') or the mean over
    post-instruction tokens ('post'). Returns np.array [n, L, d] (float32)."""
    L = n_layers(model)
    keys = {(l, site) for l in range(L)}
    out = []
    for ins in instructions:
        ids = chat_ids(tok, ins)
        _, store = forward(model, mx.array([ids]), HookSpec(capture=keys))
        if where == "last":
            vecs = [store[(l, site)][0, -1] for l in range(L)]
        else:
            pos = mx.array(post_instruction_positions(tok, ins))
            vecs = [store[(l, site)][0, pos].mean(axis=0) for l in range(L)]
        v = mx.stack(vecs).astype(mx.float32)
        mx.eval(v)
        out.append(np.array(v))
    return np.stack(out)


def diff_in_means(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    """[L, d] refusal direction per layer (not normalised)."""
    return harmful.mean(0) - harmless.mean(0)


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def cosine_per_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (unit(a) * unit(b)).sum(-1)


def subspace(harmful: np.ndarray, harmless: np.ndarray, layer: int, k: int = 3) -> np.ndarray:
    """Top-k directions separating harmful from harmless at one layer.

    Uses SVD of harmful activations minus the harmless mean, so the first
    direction is close to diff-in-means and the rest capture spread.
    Returns [k, d] orthonormal rows.
    """
    X = harmful[:, layer] - harmless[:, layer].mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return vt[:k]


def principal_angles_deg(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between row spaces of U [k,d] and V [k,d]."""
    s = np.linalg.svd(U @ V.T, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))
