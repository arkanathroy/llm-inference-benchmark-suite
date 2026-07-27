"""
evaluate.py
============
Accuracy-side evaluation used to quantify the "quality-accuracy
tradeoff" the JD explicitly requires alongside speed/memory numbers for
every quantization method.

Two complementary metrics are used deliberately, not just one:

1. Perplexity on WikiText-2 (intrinsic metric)
   WHY: perplexity is cheap to compute (single forward pass, no
   generation needed), highly sensitive to quantization-induced
   numerical degradation, and is the standard metric reported in every
   GPTQ/AWQ/LLM.int8() paper -- makes this benchmark's numbers directly
   comparable to published results as a sanity check.

2. Accuracy on a small MMLU subset (extrinsic/task metric)
   WHY: perplexity measures how well the model predicts held-out text
   token-by-token, but does NOT directly measure whether the model can
   still correctly perform a real reasoning/knowledge task. Two
   quantization methods can show near-identical perplexity while one
   causes measurably worse multiple-choice accuracy on tasks requiring
   precise numerical/logical reasoning. Testing both catches this
   failure mode.

"""

import math
from typing import List

import torch
from datasets import load_dataset


@torch.no_grad()
def compute_perplexity(model, tokenizer, max_samples: int = 50, stride: int = 512) -> float:
    """
    Sliding-window perplexity over WikiText-2's test split.

    WHY stride=512 rather than computing perplexity on independent
    short chunks: without overlap, each chunk's first few tokens are
    predicted with no context, artificially inflating perplexity.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"][:max_samples])
    encodings = tokenizer(text, return_tensors="pt")

    max_length = model.config.max_position_embeddings if hasattr(
        model.config, "max_position_embeddings"
    ) else 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end = end
        if end == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / end).item()


@torch.no_grad()
def evaluate_mmlu_subset(model, tokenizer, n_questions: int = 100, subject: str = "high_school_mathematics") -> dict:
    """
    Multiple-choice accuracy on a small MMLU subject subset.

    WHY only 100 questions and a single subject rather than the full
    MMLU: the goal is a relative comparison across 4 precision variants
    on the SAME questions, not an absolute leaderboard score. 100
    questions gives a statistically meaningful signal for detecting a
    >10% accuracy swing between quantization methods.

    WHY "high_school_mathematics" specifically: math reasoning questions
    are disproportionately sensitive to numerical precision loss
    compared to factual-recall subjects, making this subject a more
    discriminating stress test for quantization-induced degradation.
    """
    dataset = load_dataset("cais/mmlu", subject, split="test")
    dataset = dataset.select(range(min(n_questions, len(dataset))))

    correct = 0
    choices_letters = ["A", "B", "C", "D"]

    for item in dataset:
        prompt = f"{item['question']}\n"
        for letter, choice in zip(choices_letters, item["choices"]):
            prompt += f"{letter}. {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        pred_token = tokenizer.decode(outputs[0][-1]).strip().upper()
        true_letter = choices_letters[item["answer"]]

        if pred_token == true_letter:
            correct += 1

    return {
        "subject": subject,
        "n_questions": len(dataset),
        "accuracy": correct / len(dataset),
    }


def accuracy_delta_report(baseline_metrics: dict, quantized_metrics: dict) -> dict:
    """
    Computes relative degradation vs the FP16 baseline for both metrics.
    """
    ppl_delta_pct = (
        (quantized_metrics["perplexity"] - baseline_metrics["perplexity"])
        / baseline_metrics["perplexity"] * 100
    )
    acc_delta_pct = (
        (quantized_metrics["mmlu_accuracy"] - baseline_metrics["mmlu_accuracy"])
        / baseline_metrics["mmlu_accuracy"] * 100
    )
    return {
        "perplexity_increase_pct": round(ppl_delta_pct, 2),
        "mmlu_accuracy_change_pct": round(acc_delta_pct, 2),
    }
