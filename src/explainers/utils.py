import math
import torch
import torch.nn.functional as F

def align_llm_visuals_to_pixels(pixel_attribution: torch.Tensor,
                                inputs: dict) -> torch.Tensor:
    """
    Reshapes and interpolates LLM-level pixel attributions to match 
    the exact spatial footprint of the model's original pixel_values.
    
    Args:
        pixel_attribution: Tensor of shape [gen_len, num_llm_tokens]
        inputs: The original Hugging Face inputs dictionary.
        
    Returns:
        Tensor of shape [gen_len, target_num_pixels]
    """
    pixel_values = inputs.get("pixel_values")
    
    # Safety check: if there is no image, just return the raw tensor
    if pixel_values is None or pixel_attribution.numel() == 0:
        return pixel_attribution
        
    gen_len, num_llm_tokens = pixel_attribution.shape
    
    # ---------------------------------------------------------
    # CASE A: Standard VLM (e.g., LLaVA, BLIP) 
    # Shape is (C, H, W) or (Batch, C, H, W)
    # ---------------------------------------------------------
    if pixel_values.ndim >= 3 and pixel_values.shape[-3] in [1, 3, 4]:
        target_h, target_w = pixel_values.shape[-2], pixel_values.shape[-1]
        
        # Standard models usually have square ViT grids
        llm_grid_h = llm_grid_w = int(math.sqrt(num_llm_tokens))
        
        interp_mode = 'bilinear'
        align_corners = False
        
    # ---------------------------------------------------------
    # CASE B: Packed Patches VLM (e.g., Qwen-VL) 
    # Shape is (grid_h * grid_w, patch_dim)
    # ---------------------------------------------------------
    elif 'image_grid_thw' in inputs:
        _, target_h, target_w = inputs['image_grid_thw'][0].cpu().numpy().tolist()
        
        # Qwen uses a 2x2 spatial merge before the LLM
        spatial_merge_size = 2
        llm_grid_h = target_h // spatial_merge_size
        llm_grid_w = target_w // spatial_merge_size
        
        interp_mode = 'nearest'
        align_corners = None
        
    else:
        raise ValueError("Could not infer spatial grid from inputs. Unknown architecture.")

    # 1. Reshape to native 2D grid
    # Shape: [gen_len, 1, llm_grid_h, llm_grid_w]
    pixel_attr_2d = pixel_attribution.reshape(gen_len, 1, llm_grid_h, llm_grid_w)

    # 2. Spatially upsample to match pixel_values
    # Shape: [gen_len, 1, target_h, target_w]
    pixel_attr_upscaled = F.interpolate(
        pixel_attr_2d, 
        size=(target_h, target_w), 
        mode=interp_mode,
        align_corners=align_corners
    )

    # 3. Flatten back to 1D
    # Shape: [gen_len, target_h * target_w]
    return pixel_attr_upscaled.reshape(gen_len, -1)

