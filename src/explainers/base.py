from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple
import torch
import numpy as np

from src.models.base import BaseVLMWrapper # Assuming previous code is here

@dataclass
class XAIResult:
    """
    A unified container for attribution results.
    Keeps data on CPU to save GPU memory during analysis.
    """
    # The raw attribution scores
    image_attributions: torch.Tensor     # Shape: (C, H, W) or (Num_Patches,)
    text_attributions: torch.Tensor      # Shape: (Seq_Len,)
    
    # Metadata for visualization
    input_tokens: List[str]              # The actual text tokens
    pixel_values: torch.Tensor           # The original image (for plotting overlay)
    
    # Validation data
    score: float                         # The total attribution score (sum)
    target_token: str                    # What token were we explaining?

    def cpu(self):
        """Moves all tensors to CPU to avoid OOM during benchmarking."""
        self.image_attributions = self.image_attributions.detach().cpu()
        self.text_attributions = self.text_attributions.detach().cpu()
        if isinstance(self.pixel_values, torch.Tensor):
            self.pixel_values = self.pixel_values.detach().cpu()
        return self

class BaseExplainer(ABC):
    def __init__(self, model_wrapper: BaseVLMWrapper):
        self.wrapper = model_wrapper
        self.device = model_wrapper.device

    @abstractmethod
    def get_raw_attributions(self,
                             image,
                             question: str,
                             target_indices: List[int],
                             **kwargs) -> Tuple[torch.Tensor]:
        """
        Implement the specific XAI logic here (e.g., call Captum).
        Must return a tensor of shape (Batch, Seq_Len, Hidden_Dim) 
        corresponding to inputs_embeds.
        """
        pass
