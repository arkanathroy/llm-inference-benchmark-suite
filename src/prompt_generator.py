"""
prompt_generator.py
====================
Synthetic prompt generator for the load-testing sweep, implementing the
mixed short/medium/long distribution defined in
config.LocustConfig.prompt_length_mix.

WHY synthetic prompts, not a real dataset, for the load test specifically:
the load-testing goal is to control token-length distribution precisely
in order to isolate the prefill-vs-decode contention effect -- a real
dataset's length distribution is neither controllable nor reproducible
across benchmark runs. Real datasets (WikiText-2, Pile) are used instead
for accuracy evaluation and GPTQ/AWQ calibration, where content realism
matters more than length control.
"""

import random
from typing import List

from config import CONFIG

LOCUST_CFG = CONFIG["locust"]

_TOPIC_BANK = [
    "billing dispute on account", "order tracking status update",
    "password reset assistance request", "product return policy question",
    "subscription cancellation process", "technical support for device",
    "refund status inquiry follow up", "shipping address change request",
    "warranty claim documentation needed", "appointment rescheduling request",
]

_FILLER_BANK = [
    "The customer previously mentioned", "According to the account history",
    "As noted in the prior conversation", "Following up on the earlier ticket",
    "Based on the system logs shown", "The representative should verify",
    "Please confirm the details before proceeding", "This has been an ongoing issue since",
]


def _build_text(target_tokens: int, tokenizer) -> str:
    """Grows a realistic-sounding prompt until it reaches ~target_tokens
    as measured by the ACTUAL tokenizer (not word count)."""
    parts = [random.choice(_TOPIC_BANK) + "."]
    while len(tokenizer.encode(" ".join(parts))) < target_tokens:
        parts.append(random.choice(_FILLER_BANK) + " " + random.choice(_TOPIC_BANK) + ".")
    text = " ".join(parts)
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def generate_prompt_batch(n_prompts: int, tokenizer, seed: int = 42) -> List[str]:
    """
    Generates n_prompts prompts sampled from the configured length mix.
    seed=42 fixed by default so every engine/precision combination in
    the comparison table is benchmarked against the IDENTICAL prompt set.
    """
    rng = random.Random(seed)
    mix = LOCUST_CFG.prompt_length_mix
    categories = list(mix.keys())
    weights = [mix[c][2] for c in categories]

    prompts = []
    for _ in range(n_prompts):
        category = rng.choices(categories, weights=weights, k=1)[0]
        min_tok, max_tok, _ = mix[category]
        target_len = rng.randint(min_tok, max_tok)
        prompts.append(_build_text(target_len, tokenizer))
    return prompts


def prompt_length_distribution_report(prompts: List[str], tokenizer) -> dict:
    """Sanity-check utility: verifies the generated batch actually
    matches the configured mix."""
    lengths = [len(tokenizer.encode(p)) for p in prompts]
    return {
        "n_prompts": len(prompts),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "p50_tokens": sorted(lengths)[len(lengths) // 2],
    }
