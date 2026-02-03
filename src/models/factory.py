from typing import Union, Type, Optional
from transformers import (
    AutoConfig, 
    AutoProcessor, 
    AutoModelForCausalLM,
    LlavaForConditionalGeneration,
    # Qwen2_5_VLForConditionalGeneration # Import if strictly needed for type checking
)
# Fallback import if Qwen is not installed
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

from src.models import BaseVLMWrapper, QwenVL_Wrapper
from .config import LoaderConfig

class VLMType:
    # LLAVA = "llava"
    QWEN2_5_VL = "qwen2_5_vl"

def create_model_wrapper(
    model_id: str,
    vlm_type: str,
    config: Optional[LoaderConfig] = None,
) -> BaseVLMWrapper:
    
    if config is None:
        config = LoaderConfig()

    print(f"[*] Inspecting model architecture: {model_id}...")
    

    # Prepare Loading Arguments
    load_kwargs = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "dtype": config.compute_dtype,
        "quantization_config": config.get_bnb_config(),
        "attn_implementation": config.attn_implementation
    }

    # 3. Load Processor (Generic)
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=config.trust_remote_code
        )
    except Exception as e:
        # Some new models (like Qwen2.5-VL) might need specific handling if AutoProcessor fails
        # but usually AutoProcessor is sufficient.
        raise ValueError(f"Failed to load processor for {model_id}: {e}")
    
    try:
        if vlm_type == VLMType.QWEN2_5_VL:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id,
                                                                    **load_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load model for {model_id}: {e}")
    
    model.eval()
    
    # 5. Dispatch to Correct Wrapper
    if vlm_type == VLMType.QWEN2_5_VL:
        return QwenVL_Wrapper(model, processor)
    
    # elif vlm_type == VLMType.LLAVA:
    #     return LlavaWrapper(model, processor)
    
    # elif vlm_type == VLMType.INTERNVL:
    #     # InternVL usually works with LlavaWrapper logic or needs a subclass
    #     return LlavaWrapper(model, processor)

    else:
        raise NotImplementedError(f"Wrapper for {vlm_type} is defined in Enum but not instantiated in factory.")