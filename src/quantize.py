"""
quantize.py
===========
Implements the four precision variants benchmarked in this suite:
  1. FP16                - baseline "optimized" precision
  2. INT8 (bitsandbytes)  - LLM.int8() mixed-precision decomposition
  3. GPTQ (4-bit)         - post-training, calibration-based weight quant
  4. AWQ (4-bit)          - activation-aware weight quantization

See config.py QuantizationConfig for the full rationale behind every
hyperparameter referenced here.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CONFIG

MODEL_CFG = CONFIG["model"]
QUANT_CFG = CONFIG["quant"]


def load_fp16(model_id: str = None):
    """
    Baseline precision. All quantized variants are compared against this,
    not against FP32 -- FP32 is skipped entirely (see config.py docstring).
    """
    model_id = model_id or MODEL_CFG.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    return model, tokenizer


def load_int8(model_id: str = None):
    """
    LLM.int8() via bitsandbytes. See QuantizationConfig.int8_threshold
    and int8_skip_modules for full rationale on the two key parameters.
    """
    model_id = model_id or MODEL_CFG.model_id
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=QUANT_CFG.int8_threshold,
        llm_int8_skip_modules=list(QUANT_CFG.int8_skip_modules),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    model.eval()
    return model, tokenizer


def quantize_gptq(model_id: str = None, calib_texts=None, save_dir: str = "./gptq_model"):
    """
    Calibration-based GPTQ quantization using AutoGPTQ.

    IMPORTANT: unlike INT8/bitsandbytes (quantized on-the-fly at load
    time), GPTQ and AWQ both require an offline calibration pass BEFORE
    the model can be loaded for inference. This function performs that
    calibration once and saves the quantized checkpoint to disk -- the
    benchmark notebook (Phase 2) runs this once, then simply loads the
    saved artifact in every subsequent cell to avoid re-calibrating.
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    model_id = model_id or MODEL_CFG.model_id

    quantize_config = BaseQuantizeConfig(
        bits=QUANT_CFG.gptq_bits,
        group_size=QUANT_CFG.gptq_group_size,
        damp_percent=QUANT_CFG.gptq_damp_percent,
        desc_act=QUANT_CFG.gptq_desc_act,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_id, quantize_config, device_map="cuda:0"
    )

    if calib_texts is None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calib_texts = [t for t in ds["text"] if len(t.strip()) > 200][
            : QUANT_CFG.gptq_calibration_samples
        ]

    calib_data = [tokenizer(t, return_tensors="pt") for t in calib_texts]

    t0 = time.time()
    model.quantize(calib_data)
    calib_seconds = time.time() - t0

    model.save_quantized(save_dir)
    tokenizer.save_pretrained(save_dir)

    return {"save_dir": save_dir, "calibration_seconds": calib_seconds,
            "n_calib_samples": len(calib_texts)}


def load_gptq(save_dir: str = "./gptq_model"):
    from auto_gptq import AutoGPTQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoGPTQForCausalLM.from_quantized(
        save_dir,
        device_map="cuda:0",
        use_safetensors=True,
        inject_fused_attention=False,
    )
    model.eval()
    return model, tokenizer


def quantize_awq(model_id: str = None, save_dir: str = "./awq_model"):
    """
    Activation-aware Weight Quantization via AutoAWQ.
    """
    from awq import AutoAWQForCausalLM

    model_id = model_id or MODEL_CFG.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoAWQForCausalLM.from_pretrained(model_id)

    quant_config = {
        "zero_point": QUANT_CFG.awq_zero_point,
        "q_group_size": QUANT_CFG.awq_group_size,
        "w_bit": QUANT_CFG.awq_bits,
        "version": "GEMM",
    }

    t0 = time.time()
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=QUANT_CFG.awq_calib_dataset,
    )
    calib_seconds = time.time() - t0

    model.save_quantized(save_dir)
    tokenizer.save_pretrained(save_dir)

    return {"save_dir": save_dir, "calibration_seconds": calib_seconds}


def load_awq(save_dir: str = "./awq_model"):
    from awq import AutoAWQForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoAWQForCausalLM.from_quantized(save_dir, device_map="cuda:0")
    model.eval()
    return model, tokenizer


def model_memory_footprint_mb(model) -> float:
    """Reports actual allocated weight memory, not theoretical bit-width
    math -- captures real overhead from scale factors, zero-points, and
    padding that naive bits-per-param calculations miss."""
    return sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)
