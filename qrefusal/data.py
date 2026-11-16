"""Prompt sets and chat formatting."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RD = ROOT / "data" / "refusal_direction"


def split(name: str, n: int | None = None, seed: int = 0) -> list[str]:
    """Instructions from a refusal_direction split, e.g. 'harmful_test'.

    Sampling is deterministic given (name, n, seed).
    """
    rows = json.loads((RD / f"{name}.json").read_text())
    items = [r["instruction"] for r in rows]
    if n is not None and n < len(items):
        items = random.Random(f"{name}-{seed}").sample(items, n)
    return items


# Llama 3.x templates inject today's date; pin it so runs on different days match.
# "10 Nov 2026" is the date every existing Llama run used.
DATE = "10 Nov 2026"


def chat_text(tok, instruction: str) -> str:
    """Prompt string for a single user turn plus the assistant generation prompt."""
    return tok.apply_chat_template([{"role": "user", "content": instruction}],
                                   add_generation_prompt=True, tokenize=False, date_string=DATE)


def chat_ids(tok, instruction: str) -> list[int]:
    """Token ids for a single user turn plus the assistant generation prompt."""
    return tok.apply_chat_template([{"role": "user", "content": instruction}],
                                   add_generation_prompt=True, date_string=DATE)


def reply_ids(tok, text: str) -> list[int]:
    """Token ids of a reply continuation, without BOS or other special tokens."""
    return tok.encode(text, add_special_tokens=False)


def instruction_positions(tok, instruction: str) -> list[int]:
    """Positions of the user's instruction tokens (between the template prefix
    and the post-instruction suffix)."""
    full = chat_ids(tok, instruction)
    shifted = chat_ids(tok, "" + instruction)
    k = 0
    while k < min(len(full), len(shifted)) and full[k] == shifted[k]:
        k += 1
    post = post_instruction_positions(tok, instruction)
    return list(range(k, post[0] if post else len(full)))


def post_instruction_positions(tok, instruction: str) -> list[int]:
    """Positions of the chat-template tokens after the user's instruction.

    Arditi et al. find the refusal direction is most salient at these tokens.
    Computed as the suffix that the template appends after the instruction text.
    """
    full = chat_ids(tok, instruction)
    marker = ""
    with_marker = chat_ids(tok, instruction + marker)
    # The template suffix is the run of identical tokens at the end of both.
    k = 0
    while k < min(len(full), len(with_marker)) and full[-1 - k] == with_marker[-1 - k]:
        k += 1
    return list(range(len(full) - k, len(full)))


def neutralize(text: str) -> str:
    """Break chat control strings like <|im_start|> so a judge cannot read them
    as real control tokens (quantized replies sometimes emit them literally)."""
    return text.replace("<|", "<\u200b|").replace("|>", "|\u200b>")


def greedy_reply(model, tok, prompt_ids: list[int], max_tokens: int) -> tuple[str, str]:
    """Greedy continuation of token ids. Returns (text, finish_reason), where
    finish_reason is "stop" (end-of-turn/EOS emitted) or "length" (hit the cap)."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    text, reason = "", "length"
    for resp in stream_generate(model, tok, prompt=prompt_ids, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
        text += resp.text
        reason = resp.finish_reason or reason
    return text, reason


def sampled_reply(model, tok, prompt_ids: list[int], max_tokens: int, temp: float,
                  top_p: float, top_k: int, seed: int) -> tuple[str, str]:
    """Like greedy_reply but sampled, with an explicit per-call RNG seed."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k)
    text, reason = "", "length"
    for resp in stream_generate(model, tok, prompt=prompt_ids, max_tokens=max_tokens, sampler=sampler):
        text += resp.text
        reason = resp.finish_reason or reason
    return text, reason
