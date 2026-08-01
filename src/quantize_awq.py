"""Quantize the configured base model to AWQ using LLM Compressor.

Runs inside envs/awq/venv.

Migrated from the deprecated `autoawq` package (see upstream deprecation
notice: https://github.com/vllm-project/llm-compressor) to LLM Compressor's
AWQModifier, which is the vLLM project's actively-maintained successor to
AutoAWQ. Functionally equivalent (same AWQ algorithm, same calibration-based
scale search), but built on `compressed-tensors` and integrated with vLLM's
own quantization runtime rather than a frozen standalone library.

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
Unlike a hardcoded scheme preset string (e.g. "W4A16"), this builds the
QuantizationScheme explicitly from AWQConfig.bits/group_size/zero_point so
those fields keep actually controlling the quantization -- preserving the
apples-to-apples bit-width/group-size parity with GPTQConfig that this
comparison depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG


def build_calibration_dataset(tokenizer, num_samples=512, max_seq_length=512):
    """Same wikitext calibration source used by quantize_gptq.py, kept
    identical here so GPTQ and AWQ are calibrated on matched data for a
    fair apples-to-apples comparison between the two techniques.
    """
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)


def main():
    model_id = CONFIG.model.hf_repo
    awq_cfg = CONFIG.awq
    output_dir = Path(awq_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

    calibration_dataset = build_calibration_dataset(tokenizer)

    # Built directly from AWQConfig fields -- bits/group_size/zero_point
    # from config.py all take effect here, matching GPTQConfig's bits and
    # group_size exactly as intended for the fair-comparison framing.
    weight_args = QuantizationArgs(
        num_bits=awq_cfg.bits,
        symmetric=not awq_cfg.zero_point,
        strategy=QuantizationStrategy.GROUP,
        group_size=awq_cfg.group_size,
    )
    quant_scheme = QuantizationScheme(targets=["Linear"], weights=weight_args)

    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            config_groups={"group_0": quant_scheme},
        ),
    ]

    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=512,
        num_calibration_samples=len(calibration_dataset),
    )

    model.save_pretrained(str(output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(output_dir))

    print(
        f"AWQ-quantized model (via llm-compressor) written to {output_dir} "
        f"[bits={awq_cfg.bits}, group_size={awq_cfg.group_size}, "
        f"zero_point={awq_cfg.zero_point}]"
    )


if __name__ == "__main__":
    main()