import torch
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper


class RandomExplainer(BaseExplainer):
    """
    A baseline explainer that returns random attributions.
    Used to validate that faithfulness metrics correctly penalize random noise.
    """
    def __init__(self, model_wrapper: BaseVLMWrapper):
        super().__init__(model_wrapper)

    def get_raw_attributions(self,
                             image,
                             text: str,
                             target_indices: Optional[int | List[int]] = None,
                             seed: int = 42,
                             **kwargs
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates random attention maps for Image and Text.
        
        Args:
            image: The input image (Tensor or PIL, handled by wrapper).
            text: The text prompt.
            target_indices: Not used for random, but kept for API consistency.
            seed: Set a seed for reproducibility.
            
        Returns:
            pixel_attributions: (B, H, W) or (B, Num_Patches)
            token_attributions: (B, Seq_Len)
        """
        # 1. Prepare Inputs using the wrapper's processor
        # We need to know the shapes the model expects
        inputs = self.wrapper.get_inputs(image, text)

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]  #.unsqueeze(0)

        if pixel_values.ndim == 2: # batch is missing
            pixel_values = pixel_values.unsqueeze(0)
        
        # batch_size = input_ids.shape[0]

        pred_results = kwargs.get("pred_results", None)
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs=inputs,
                                            return_logits=False,
                                            **kwargs,
                                            )
        new_ids = pred_results["new_ids"]
        new_ids_len = len(new_ids)

        # 2. Set Seed for Reproducibility
        # We use a generator to avoid affecting the global RNG state
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # 3. Generate Random Token Attributions
        # Shape: (B, Seq_Len) - One score per token
        seq_len = input_ids.shape[1]
        token_attributions = torch.rand(
            (new_ids_len, seq_len), 
            device=self.device, 
            generator=generator
        )

        # 4. Generate Random Pixel Attributions
        # We need to decide if we return (H, W) or (Num_Patches).
        # Standard metrics usually expect spatial maps (H, W).
        
        if pixel_values.ndim == 4: # (B, C, H, W)
            _, _, h, w = pixel_values.shape
            # Generate random heatmap (B, H, W)
            pixel_attributions = torch.rand(
                (new_ids_len, h, w), 
                device=self.device, 
                generator=generator
            )
        elif pixel_values.ndim == 3: # (B, num_patches, patch_dim)
            _, num_patches, _ = pixel_values.shape
            try:
                _, h, w = inputs["image_grid_thw"][0].cpu().numpy().tolist()
                pixel_attributions = torch.rand(
                    (new_ids_len, h*w), 
                    device=self.device, 
                    generator=generator
                )
            except _:  
                pixel_attributions = torch.rand(
                    (new_ids_len, num_patches), 
                    device=self.device, 
                    generator=generator
                )
        else:
            raise ValueError("pixel_values shape must be (B, C, H, W) or (B, num_patches, patch_dim).")

        return token_attributions, pixel_attributions
