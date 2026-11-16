"""Identity tests that must pass before any experiment run.

Uses a small model so the suite runs in under a minute:
    QREF_TEST_FP = bf16 model path or HF id (default Qwen/Qwen2.5-0.5B-Instruct)
    QREF_TEST_Q  = local quantized copy (default models/qwen2.5-0.5b-4bit-g64)
"""

import os

import mlx.core as mx
import pytest

from qrefusal.hooks import HookSpec, forward, n_layers
from qrefusal.models import GROUPS, load, restore

FP = os.environ.get("QREF_TEST_FP", "Qwen/Qwen2.5-0.5B-Instruct")
Q = os.environ.get("QREF_TEST_Q", "models/qwen2.5-0.5b-4bit-g64")


@pytest.fixture(scope="module")
def pair():
    f, tok = load(FP)
    q, _ = load(Q)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": "How do I bake bread?"}], add_generation_prompt=True
    )
    return f, q, mx.array([ids])


def maxdiff(a, b):
    return mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()


def test_forward_matches_model(pair):
    f, _, x = pair
    ref = f(x)
    out, _ = forward(f, x)
    assert maxdiff(ref, out) < 1e-3


def test_patch_self_is_noop(pair):
    f, _, x = pair
    keys = {(L, s) for L in range(n_layers(f)) for s in ("resid_post", "attn", "mlp")}
    base, acts = forward(f, x, HookSpec(capture=keys))
    patched, _ = forward(f, x, HookSpec(patch={k: (None, v) for k, v in acts.items()}))
    assert maxdiff(base, patched) < 1e-3


def test_add_zero_is_noop(pair):
    f, _, x = pair
    base, acts = forward(f, x, HookSpec(capture={(5, "resid_post")}))
    d = acts[(5, "resid_post")].shape[-1]
    out, _ = forward(f, x, HookSpec(add={(5, "resid_post"): ([x.shape[1] - 1], mx.zeros((d,)))}))
    assert maxdiff(base, out) < 1e-3


def test_full_restoration_equals_fp(pair):
    f, q, x = pair
    r = restore(q, f, range(n_layers(q)), GROUPS.keys(), io=True)
    assert maxdiff(f(x), forward(r, x)[0]) < 1e-2


def test_restore_does_not_mutate_quantized(pair):
    f, q, x = pair
    before = forward(q, x)[0]
    _ = restore(q, f, [0, 1], ["mlp"])
    assert maxdiff(before, forward(q, x)[0]) == 0.0
