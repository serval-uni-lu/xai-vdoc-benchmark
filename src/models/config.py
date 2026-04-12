from dataclasses import dataclass

import torch
from transformers import BitsAndBytesConfig


@dataclass
class LoaderConfig:
    """Configuration for loading the generic VLM."""

    device_map: str = "cuda:0"  # Change to auto in benchmark
    load_in_4bit: bool = True
    compute_dtype: torch.dtype = torch.bfloat16
    attn_implementation: str | None = None  # e.g. "flash_attention_2" or "eager"
    trust_remote_code: bool = True

    def get_bnb_config(self) -> BitsAndBytesConfig | None:
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                # bnb_4bit_quant_type="nf4", # Usually standard
            )
        return None
