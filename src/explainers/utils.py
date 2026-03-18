import math
from typing import List, Optional, Dict, Any
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from PIL import Image
from captum.attr import visualization
from matplotlib.colors import LinearSegmentedColormap

class XAIVisualizer:
    def __init__(self, processor):
        """
        Args:
            processor: The Hugging Face processor/tokenizer.
        """
        self.processor = processor

    def _get_cmap(self):
        cmap = LinearSegmentedColormap.from_list(
            'red_blue',
            [
                (0.0, '#0000ff'),  # blue   (low values)
                (0.5, '#ffffff'),  # white  (mid)
                (1.0, '#ff0000'),  # red    (high values)
            ],
            N=256,
        )
        return cmap

    def plot_text_attributions(self, 
                               text_attr: torch.Tensor, 
                               input_ids: torch.Tensor, 
                               target_ids: torch.Tensor,
                               special_token_ids: Optional[List] = None,
                               semantic_mask: Optional[torch.Tensor] = None,
                               normalize: bool = True):
        """
        Visualizes text attributions, filtering out visual/special tokens.
        
        Args:
            text_attr: Shape (num_answer_tokens, seq_len)
            input_ids: Shape (1, seq_len)
            target_ids: Shape (1, num_answer_tokens)
            special_token_ids: List of token IDs to exclude (e.g., <|image_pad|>)
            normalize: Whether to scale scores to [-1, 1] for better color contrast
        """
        special_token_ids = special_token_ids or []
        input_ids_list = input_ids[0].tolist()
        
        # 1. Identify valid tokens
        valid_indices = []
        for i, tok_id in enumerate(input_ids_list):
            # Condition A: Always drop pure visual/padding tokens
            if tok_id in special_token_ids:
                continue
                
            # Condition B: If a semantic mask exists, drop tokens where it is False
            if semantic_mask is not None:
                if not semantic_mask[0, i].item():
                    continue
            
            valid_indices.append(i)
            
        if len(valid_indices) == 0:
            print("[!] Warning: No valid text tokens found to visualize.")
            return
        
        # 2. Slice the input_ids and the attributions to keep ONLY valid text
        filtered_input_ids = input_ids[0][valid_indices]
        filtered_text_attr = text_attr[:, valid_indices]
        
        # 3. Decode the remaining valid tokens for Captum
        # Using batch_decode on individual tokens to preserve exact 1-to-1 mapping
        tokens = self.processor.batch_decode(
            filtered_input_ids.unsqueeze(1), # batch_decode expects a list of sequences
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
        # Flatten the list of lists returned by batch_decode
        if isinstance(tokens[0], list):
             tokens = [t[0] for t in tokens]
        
        records = []
        num_answer_tokens = filtered_text_attr.shape[0]

        for i in range(num_answer_tokens):
            attr_scores = filtered_text_attr[i].cpu().detach().numpy()
            
            # --- NORMALIZATION ---
            if normalize:
                max_abs = np.max(np.abs(attr_scores))
                if max_abs > 0:
                    attr_scores = attr_scores / max_abs # Scale to [-1, 1]
            
            # Decode the target token being explained
            target_token_str = self.processor.decode([target_ids[0, i]])
            
            record = visualization.VisualizationDataRecord(
                word_attributions=attr_scores,
                pred_prob=1.0, 
                pred_class=target_token_str, 
                true_class=target_token_str, 
                attr_class=target_token_str,
                attr_score=attr_scores.sum(),
                raw_input_ids=tokens,
                convergence_score=0.0
            )
            records.append(record)
            
        print("\n" + "="*50)
        print("TEXT ATTRIBUTIONS (Filtered & Normalized)")
        print("="*50)
        _ = visualization.visualize_text(records)

    def plot_image_attributions(self, 
                                img_attr: Tensor, 
                                original_image: Image.Image, 
                                target_ids: Tensor, 
                                image_grid_thw: Optional[Tensor] = None):
        """
        Visualizes image attributions dynamically, handling both flattened patches and 2D maps.
        
        Args:
            img_attr: Shape (num_answer_tokens, num_patches) OR (num_answer_tokens, H, W)
            original_image: The raw PIL Image object.
            target_ids: Shape (1, num_answer_tokens)
            image_grid_thw: Optional. Shape (1, 3). Required if img_attr is flattened patches (like Qwen).
        """
        num_tokens = img_attr.shape[0]
        orig_w, orig_h = original_image.size
        
        # 1. Reshape Attributions to 2D Spatial Grid if necessary
        if img_attr.dim() == 2: # Shape: (num_tokens, num_patches)
            num_patches = img_attr.shape[1]
            
            if image_grid_thw is not None:
                # Qwen-style dynamic grid
                grid_h = image_grid_thw[0, 1].item()
                grid_w = image_grid_thw[0, 2].item()
                assert grid_h * grid_w == num_patches, f"Grid {grid_h}x{grid_w} != {num_patches} patches"
            else:
                # Fallback: Assume square grid (e.g., standard ViT)
                grid_h = grid_w = int(np.sqrt(num_patches))
                
            # Reshape: (num_tokens, grid_h, grid_w)
            attrs_2d = img_attr.view(num_tokens, grid_h, grid_w)
            
        elif img_attr.dim() == 3: # Already (num_tokens, H, W)
            attrs_2d = img_attr
        else:
            raise ValueError(f"Unexpected img_attr shape: {img_attr.shape}")

        # 2. Upsample Heatmap to exactly match the PIL Image dimensions
        attrs_upsampled = F.interpolate(
            attrs_2d.unsqueeze(1).float(), # (num_tokens, 1, H, W)
            size=(orig_h, orig_w),         # Target size
            mode='bilinear', 
            align_corners=False
        ).squeeze(1) # (num_tokens, orig_h, orig_w)

        # 3. Prepare the Background Canvas (PIL -> Numpy -> Normalize to [0,1])
        rgb_background = np.array(original_image.convert("RGB")).astype(np.float32) / 255.0

        # 4. Plot using Captum
        print("\n" + "="*50)
        print("IMAGE ATTRIBUTIONS")
        print("="*50)
        
        cmap = self._get_cmap()

        for i in range(num_tokens):
            target_token_str = self.processor.decode([target_ids[0, i]])
            print(f"\n[*] Heatmap for generated token: '{target_token_str}'")
            
            # Captum requires the heatmap to have a channel dim: (H, W, 1)
            attr_map = attrs_upsampled[i].unsqueeze(-1).cpu().detach().numpy()
            
            _ = visualization.visualize_image_attr_multiple(
                attr_map,
                rgb_background,
                methods=["original_image", "heat_map", "blended_heat_map"],
                signs=["all", "positive", "positive"],
                show_colorbar=True,
                titles=["Original Image", f"Attribution: '{target_token_str}'", "Overlay"],
                use_pyplot=True,
                cmap=cmap,
            )

def align_llm_visuals_to_pixels(pixel_attribution: Tensor,
                                inputs: dict) -> Tensor:
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

def align_attribution_to_patches(
    high_res_attr: torch.Tensor, 
    image_grid_thw: torch.Tensor
) -> torch.Tensor:
    """
    Downsamples a pixel-level attribution map to match the VLM's patch sequence.
    
    Args:
        high_res_attr: Shape (num_answer_tokens, H, W) or (H, W). The raw attribution.
        image_grid_thw: Shape (1, 3). The temporal/height/width grid from Qwen.
        
    Returns:
        patch_attr: Shape (num_answer_tokens, num_patches). 
                    Ready for your faithfulness metric!
    """
    # 1. Handle dimensionality
    if high_res_attr.dim() == 2:
        high_res_attr = high_res_attr.unsqueeze(0) # (1, H, W)
        
    num_tokens = high_res_attr.shape[0]
    
    # 2. Extract the target grid dimensions from Qwen's metadata
    # image_grid_thw is usually [temporal (1), grid_h, grid_w]
    grid_h = image_grid_thw[0, 1].item()
    grid_w = image_grid_thw[0, 2].item()
    num_patches = grid_h * grid_w

    # 3. Add dummy channel dim for PyTorch pooling functions: (N, C, H, W)
    attr_4d = high_res_attr.unsqueeze(1).float() 
    
    # 4. Downsample to the Patch Grid
    # Adaptive Average Pooling is perfect here because it guarantees the output 
    # will be exactly (grid_h, grid_w), regardless of the input (H, W).
    # It effectively asks: "What is the average importance of the pixels inside this patch?"
    patch_grid_attr = F.adaptive_avg_pool2d(attr_4d, output_size=(grid_h, grid_w))
    
    # 5. Flatten back to the sequence format Qwen expects
    # (N, 1, grid_h, grid_w) -> (N, grid_h * grid_w) -> (N, num_patches)
    patch_attr_flat = patch_grid_attr.view(num_tokens, num_patches)
    
    return patch_attr_flat

def create_semantic_mask_robust(input_ids: Tensor,
                                processor,
                                prefix_text: str,
                                core_question: str) -> Tensor:
    """
    Finds the exact tokens corresponding to the core question using cumulative decoding.
    This safely ignores BPE space-merging and special inserted tokens.
    
    Args:
        input_ids: (1, Seq_Len) tensor
        processor: The model's processor/tokenizer
        prefix_text: The instruction text BEFORE the question (e.g., "Answer strictly...")
        core_question: The question itself
    """
    seq_len = input_ids.shape[1]
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=input_ids.device)
    
    # Clean strings to avoid trailing whitespace mismatches
    prefix_clean = prefix_text.strip()
    target_clean = (prefix_text + core_question).strip()
    
    start_idx = None
    end_idx = None
    
    for i in range(1, seq_len + 1):
        # Decode the sequence up to the current token
        # skip_special_tokens=True ignores <|im_start|>, <|image_pad|>, etc.
        decoded_so_far = processor.decode(input_ids[0, :i], skip_special_tokens=True)
        
        # --- 1. Find where the Question Starts ---
        if start_idx is None and prefix_clean in decoded_so_far:
            # Check if the current token ended EXACTLY at the prefix.
            if decoded_so_far.endswith(prefix_clean):
                start_idx = i  # The NEXT token (index i) starts the question
            else:
                # The current token bridged the gap (e.g. it contains "no: Is")
                start_idx = i - 1 
                
        # --- 2. Find where the Question Ends ---
        if start_idx is not None and target_clean in decoded_so_far:
            end_idx = i - 1
            break
            
    if start_idx is not None and end_idx is not None:
        mask[0, start_idx : end_idx + 1] = True
    else:
        print(f"[!] Warning: Could not align semantic mask for: '{core_question}'")
        # Fallback: mask all text (or handle as needed)
        
    return mask

def create_semantic_mask_robust_(input_ids: torch.Tensor,
                                processor,
                                core_question: str) -> torch.Tensor:
    """
    Finds the exact tokens corresponding to the core question by searching for the 
    tightest token window that decodes to contain the question.
    This safely ignores BPE space-merging and hidden chat template tags.
    """
    seq_len = input_ids.shape[1]
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=input_ids.device)
    
    target_clean = core_question.strip()
    
    best_i = None
    best_j = None
    min_window = float('inf')
    
    # Brute force search: O(N^2) but extremely fast for short VQA sequences
    for i in range(seq_len):
        for j in range(i + 1, seq_len + 1):
            # Decode the current slice
            decoded = processor.decode(input_ids[0, i:j], skip_special_tokens=True)
            
            # Does this slice contain our full question?
            if target_clean in decoded:
                window_size = j - i
                
                # We want the tightest possible bounds around the question
                if window_size < min_window:
                    min_window = window_size
                    best_i = i
                    best_j = j
                
                # Once we find it for this 'i', making 'j' larger just adds trailing 
                # context we don't want, so we break and move to the next 'i'
                break 
                
    if best_i is not None and best_j is not None:
        mask[0, best_i : best_j] = True
    else:
        print(f"[!] Warning: Could not align semantic mask for: '{core_question}'")
        # Fallback: Mask all non-special tokens
        mask = torch.ones_like(mask)
        
    return mask
