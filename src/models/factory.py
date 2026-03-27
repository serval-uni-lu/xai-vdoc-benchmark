from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig
)

from src.models import (
    BaseVLMWrapper,
    QwenVLWrapper,
    InternVLWrapper,
    LlavaWrapper
)

class VLMType:
    QWEN2_5_VL = "qwen2_5_vl"
    INTERNVL = "internvl"
    LLAVA = "llava"


def load_vlm(
    model_config: Dict[str, Any],
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
        "trust_remote_code": trust_remote,
        "dtype": torch.bfloat16,
        "quantization_config": bnb_config,
        "attn_implementation": attn_implementation,
    }

    # 3. Load Processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=trust_remote
        )
    except Exception as e:
        raise ValueError(f"Failed to load processor for {model_id}: {e}")
    
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
        raise ValueError(f"Failed to load model for {model_id}: {e}")
    
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

# def create_model_wrapper(
#     model_id: str,
#     vlm_type: str,
#     config: Optional[LoaderConfig] = None,
# ) -> BaseVLMWrapper:
    
#     if config is None:
#         config = LoaderConfig()

#     print(f"[*] Inspecting model architecture: {model_id}...")
    

#     # Prepare Loading Arguments
#     load_kwargs = {
#         "device_map": config.device_map,
#         "trust_remote_code": config.trust_remote_code,
#         "dtype": config.compute_dtype,
#         "quantization_config": config.get_bnb_config(),
#         "attn_implementation": config.attn_implementation
#     }

#     # Load Processor (Generic)
#     try:
#         processor = AutoProcessor.from_pretrained(
#             model_id, 
#             trust_remote_code=config.trust_remote_code
#         )
#     except Exception as e:
#         # Some new models (like Qwen2.5-VL) might need specific handling if AutoProcessor fails
#         # but usually AutoProcessor is sufficient.
#         raise ValueError(f"Failed to load processor for {model_id}: {e}")
    
#     try:
#         if vlm_type == VLMType.QWEN2_5_VL:
#             model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,
#                                                                     **load_kwargs)
#         elif vlm_type == VLMType.INTERNVL3_5:
#             model = AutoModelForImageTextToText.from_pretrained(model_id,
#                                                                 **load_kwargs)
#         elif vlm_type == VLMType.LLAVA:
#             model = LlavaForConditionalGeneration.from_pretrained(model_id,
#                                                                 **load_kwargs)
#         else:
#             model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
#     except Exception as e:
#         raise ValueError(f"Failed to load model for {model_id}: {e}")
    
#     model.eval()
    
#     # 5. Dispatch to Correct Wrapper
#     if vlm_type == VLMType.QWEN2_5_VL:
#         return QwenVLWrapper(model, processor)
    
#     elif vlm_type == VLMType.INTERNVL3_5:
#         return InternVLWrapper(model, processor)
    
#     elif vlm_type == VLMType.LLAVA:
#         return LlavaWrapper(model, processor)

#     else:
#         raise NotImplementedError(f"Wrapper for {vlm_type} is defined in Enum but not instantiated in factory.")
    
