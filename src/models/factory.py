from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from src.models import BaseVLMWrapper, InternVLWrapper, LlavaWrapper, QwenVLWrapper


class VLMType:
    QWEN2_5_VL = "qwen2_5_vl"
    INTERNVL = "internvl"
    LLAVA = "llava"


def load_vlm(
    model_config: dict[str, Any],
    attn_implementation: str | None = None,
    gpu_node: int = 0,
    output_attentions: bool = False,
) -> BaseVLMWrapper:
    """
    Unified factory to load a VLM and wrap it based on a YAML configuration.
    """
    model_id = model_config["model_id"]
    vlm_type = model_config.get("vlm_type", model_config["name"])
    trust_remote = model_config.get("trust_remote_code", False)

    print(f"[*] Loading model architecture: {model_id} on GPU {gpu_node}...")

    # 1. Setup Quantization (Equivalent to your LoaderConfig)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Prepare Loading Arguments
    load_kwargs = {
        "device_map": f"cuda:{gpu_node}",
        # "device_map": "auto",
        "trust_remote_code": trust_remote,
        "dtype": torch.bfloat16,
        "quantization_config": bnb_config,
        "attn_implementation": attn_implementation,
    }

    # 3. Load Processor
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote)
        if vlm_type == VLMType.INTERNVL:
            # Reduce the number of tiles to avoid VRAM and GPU overcomsumption
            processor.image_processor.max_patches = 2

    except Exception as e:
        raise ValueError(f"Failed to load processor for {model_id}: {e}") from e

    # 4. Load Model
    try:
        if vlm_type in [VLMType.QWEN2_5_VL, "qwenvl"]:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        elif vlm_type == VLMType.INTERNVL:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        elif vlm_type == VLMType.LLAVA:
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        else:
            print(f"[!] Warning: Falling back to AutoModelForCausalLM for {vlm_type}")
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load model for {model_id}: {e}") from e

    model.eval()

    # 5. Handle Attention Flags (Specifically for Rollout on LLaVA)
    if output_attentions and vlm_type == VLMType.LLAVA:
        model.config.vision_config.output_attentions = True
        model.vision_tower.config.output_attentions = True
        model.vision_tower.vision_model.config.output_attentions = True

    # 6. Dispatch to Correct Wrapper
    if vlm_type in [VLMType.QWEN2_5_VL, "qwenvl"]:
        return QwenVLWrapper(model, processor)

    elif vlm_type == VLMType.INTERNVL:
        return InternVLWrapper(model, processor)

    elif vlm_type == VLMType.LLAVA:
        return LlavaWrapper(model, processor)

    else:
        raise NotImplementedError(f"Wrapper for {vlm_type} is defined but not instantiated in factory.")
