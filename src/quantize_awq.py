"""Quantize the configured base model to AWQ using AutoAWQ.

Runs inside envs/awq/venv.

Reads all hyperparameters from src/config.py's CONFIG singleton — see
that module for WHAT/WHY/EFFECT documentation on every value used here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG


def main():
    model_id = CONFIG.model.hf_repo
    awq_cfg = CONFIG.awq
    output_dir = Path(awq_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_config = {
        "zero_point": awq_cfg.zero_point,
        "q_group_size": awq_cfg.group_size,
        "w_bit": awq_cfg.bits,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"AWQ-quantized model written to {output_dir}")


if __name__ == "__main__":
    main()
