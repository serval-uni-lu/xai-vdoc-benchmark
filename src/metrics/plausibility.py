import numpy as np
import cv2
from typing import Union, List, Tuple

def compute_pointing_game(
    heatmap: np.ndarray, 
    ground_truth: Union[np.ndarray, List[int]], 
    mode: str = 'mask'
) -> int:
    """
    Computes the Pointing Game metric (Hit/Miss).
    
    Args:
        heatmap: (H_map, W_map) float array. The raw attention/saliency map.
        ground_truth: 
            - If mode='mask': (H_img, W_img) binary array (1 for object, 0 for background).
            - If mode='box': List [x_min, y_min, x_max, y_max] coordinates.
        mode: 'mask' or 'box'.
        
    Returns:
        1 if the max point falls inside the ground truth, 0 otherwise.
    """
    # 1. Handle Dimensions
    # We must resize the heatmap to match the ground truth resolution
    # (unless ground_truth is just box coordinates, in which case we map coordinates)
    
    if mode == 'mask':
        assert(isinstance(ground_truth, np.ndarray))
        H_gt, W_gt = ground_truth.shape
        # Resize heatmap to match mask
        if heatmap.shape != (H_gt, W_gt):
            # Use INTER_CUBIC or LINEAR for smooth gradients
            heatmap_resized = cv2.resize(heatmap, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap
            
        # 2. Find Maximum Point
        # argmax returns the flattened index, unravel_index converts it to (y, x)
        max_y, max_x = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        
        # 3. Check Hit
        # If the pixel at (max_y, max_x) in the mask is > 0, it's a Hit.
        is_hit = ground_truth[max_y, max_x] > 0
        
    elif mode == 'box':
        assert(isinstance(ground_truth, list))
        # ground_truth is [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = ground_truth
        
        # We need to know the image size to map the heatmap coordinates roughly, 
        # OR we just assume the box is scaled to the heatmap's size.
        # Usually, it's safer to resize the heatmap to the original image size 
        # (if you know it) before checking. 
        # Assuming here that the box is in the same coordinate space as the heatmap 
        # OR that we resize heatmap to the image dimensions W_img, H_img 
        # (which requires passing those dimensions).
        
        # Simplified: Check if max point (normalized) falls in normalized box
        H_map, W_map = heatmap.shape
        max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Check standard point-in-box
        # Note: If box coordinates are for the full HR image, 
        # you must scale max_x/max_y up.
        # Here assuming standard mapping:
        
        # Let's assume you resize heatmap to the image size first (Best Practice)
        # Placeholder for resizing logic if image dims provided...
        
        is_hit = (x_min <= max_x <= x_max) and (y_min <= max_y <= y_max)
    
    else:
        raise ValueError(f"Mode {mode} is not implemented/recognized.")

    return 1 if is_hit else 0


def compute_energy_score(heatmap, segmentation_mask):
    """
    Computes what fraction of the heatmap's energy falls inside the object mask.
    Args:
        heatmap: (H, W) float array, raw attention scores.
        segmentation_mask: (H, W) binary array (1 for object, 0 for background).
    """
    # 1. Normalize heatmap to sum to 1 (Probabilistic interpretation)
    # Add epsilon to avoid division by zero
    total_energy = heatmap.sum() + 1e-9
    heatmap_norm = heatmap / total_energy
    
    # 2. Mask the heatmap
    # We only sum the energy where mask == 1
    energy_inside = (heatmap_norm * segmentation_mask).sum()
    
    return energy_inside

def energy_point_game(mask_gt, saliency_map):
    assert (mask_gt.shape == saliency_map.shape)
    
    # Enforce non-negative energy (Standard practice)
    # Use np.abs(saliency_map) if negative values indicate importance
    # Use np.maximum(saliency_map, 0) if negative values indicate suppression
    pos_saliency = np.maximum(saliency_map, 0) 
    
    mask_bbox = pos_saliency * mask_gt
    energy_bbox = mask_bbox.sum()
    energy_whole = pos_saliency.sum()
    
    # Handle the completely empty map edge case explicitly
    if energy_whole == 0:
        return 0.0
        
    return float(energy_bbox / energy_whole)

def point_game(mask_gt, salie ncy_map):
    assert (mask_gt.shape == saliency_map.shape)
    
    # Catch the blank map bug
    if saliency_map.max() == saliency_map.min():
        return 0 # If the map is completely uniform (all 0s, all 1s), it's a miss
        
    mask_bbox = saliency_map * mask_gt
    
    # Use the fast max() trick, but safely
    if mask_bbox.max() == saliency_map.max():
        # Double check that the max value isn't 0.0 created by the mask background
        # (This protects against the negative map bug)
        if saliency_map.max() > 0 or mask_gt[saliency_map == saliency_map.max()].any():
            return 1
            
    return 0
