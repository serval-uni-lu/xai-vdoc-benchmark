import torch
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper

class RandomExplainer(BaseExplainer):
    """
    A baseline explainer that returns random attributions.
    Used to validate that faithfulness metrics correctly penalize random noise.
    Compatible with Standard 4D, QwenVL 3D (patches), and InternVL 5D (tiles).
    """
    def __init__(self, model_wrapper: 'BaseVLMWrapper'):
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
            token_attributions: (new_ids_len, Seq_Len)
            pixel_attributions: (new_ids_len, H, W) or (new_ids_len, num_patches) or (new_ids_len, num_tiles, H, W)
        """
        # 1. Prepare Inputs using the wrapper's processor
        inputs = self.wrapper.get_inputs(image, text)

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]

        # --- SAFELY ADD BATCH DIMENSION BASED ON ARCHITECTURE ---
        model_type = getattr(self.wrapper.model.config, "model_type", "").lower()
        
        if "internvl" in model_type:
            # InternVL expects 5D: (Batch, num_tiles, C, H, W)
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(0)
                
        elif "qwen" in model_type:
            # QwenVL expects 3D: (Batch, num_patches, patch_dim)
            if pixel_values.ndim == 2:
                pixel_values = pixel_values.unsqueeze(0)
                
        else:
            # Fallback for Standard VLMs like LLaVA 
            # LLaVA expects 4D: (Batch, C, H, W)
            # Processors usually return 4D, so we ONLY unsqueeze if it's oddly 3D
            if pixel_values.ndim == 3: 
                pixel_values = pixel_values.unsqueeze(0)

        # Retrieve or generate predictions
        pred_results = kwargs.get("pred_results", None)
        if pred_results is None:
            pred_results = self.wrapper.predict(
                inputs=inputs,
                return_logits=False,
                **kwargs,
            )
            
        # We need the length of the generated answer to know how many heatmaps to create
        new_ids = pred_results["new_ids"]
        if new_ids.dim() > 1:
            new_ids = new_ids[0] # Handle batch dim if present
        new_ids_len = len(new_ids)

        # 2. Set Seed for Reproducibility
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # 3. Generate Random Token Attributions
        seq_len = input_ids.shape[1]
        token_attributions = torch.rand(
            (new_ids_len, seq_len), 
            device=self.device, 
            generator=generator
        )

        # 4. Generate Random Pixel Attributions based on model architecture
        ndim = pixel_values.ndim
        
        if ndim == 5: 
            # --- InternVL (AnyRes Tiling) ---
            # pixel_values shape: (B, num_tiles, C, H, W)
            _, num_tiles, _, h, w = pixel_values.shape
            pixel_attributions = torch.rand(
                (new_ids_len, num_tiles, h, w), 
                device=self.device, 
                generator=generator
            )
            
        elif ndim == 4: 
            # --- Standard CNN/ViT (LLaVA) ---
            # pixel_values shape: (B, C, H, W)
            _, _, h, w = pixel_values.shape
            pixel_attributions = torch.rand(
                (new_ids_len, h, w), 
                device=self.device, 
                generator=generator
            )
            
        elif ndim == 3: 
            # --- Qwen2-VL (Flattened Dynamic Patches) ---
            # pixel_values shape: (B, num_patches, patch_dim)
            _, num_patches, _ = pixel_values.shape
            
            if "image_grid_thw" in inputs:
                # If we have the grid, shape it to the exact 2D feature map size
                # grid_thw is usually [Temporal, Height, Width]
                grid_thw = inputs["image_grid_thw"][0].cpu().numpy().tolist()
                h, w = grid_thw[1], grid_thw[2]
                pixel_attributions = torch.rand(
                    (new_ids_len, h * w), 
                    device=self.device, 
                    generator=generator
                )
            else:
                # Absolute fallback
                pixel_attributions = torch.rand(
                    (new_ids_len, num_patches), 
                    device=self.device, 
                    generator=generator
                )
        else:
            raise ValueError(f"pixel_values shape {pixel_values.shape} is not supported.")

        return token_attributions, pixel_attributions
    
