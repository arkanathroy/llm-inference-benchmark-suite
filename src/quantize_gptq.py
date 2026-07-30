"""Quantize the configured base model to GPTQ using GPTQModel.

Runs inside envs/gptq/venv. GPTQModel is the actively maintained
successor to AutoGPTQ (AutoGPTQ's own maintainers point users to it).

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG


def build_calibration_samples(tokenizer, n_samples=128):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t.strip()) > 200][:n_samples]
    return texts


def main():
    model_id = CONFIG.model.hf_repo
    gptq_cfg = CONFIG.gptq
    output_dir = Path(gptq_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    calibration_texts = build_calibration_samples(tokenizer)

    quantize_config = QuantizeConfig(
        bits=gptq_cfg.bits,
        group_size=gptq_cfg.group_size,
        desc_act=gptq_cfg.desc_act,
    )

    model = GPTQModel.load(model_id, quantize_config)
    model.quantize(calibration_texts)
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"GPTQ-quantized model written to {output_dir}")


if __name__ == "__main__":
    main()
