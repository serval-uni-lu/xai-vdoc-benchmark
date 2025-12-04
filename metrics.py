import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Sequence, Dict, Optional, Callable


@torch.no_grad()
def eval_image_perturbation_batch(
    model_wrapper,
    # inpts,
    pixel_values: Tensor,
    target_idx: Tensor,           # (B,) tensor of class indices
    pixel_attribution: Tensor,    # (B, H, W) or (B, 1, H, W) or (B, C, H, W)
    perturbation_steps: Sequence[float],
    forward_fn: Callable[[Tensor], Tensor],
    mask_value: float = 0.0,
    descending: bool = True,      # True: most important first; False: least important
    use_sigmoid: bool = False, # Use sigmoid to compute the probs - False use log_softmax
) -> Dict[str, np.ndarray]:
    """
    Batch-level image perturbation evaluation.

    Args
    ----
    model_wrapper: nn.Module
        Wrapper with:
            - .text_embeddings_layer(input_ids)
            - __call__(text_embeds, pixel_values, pixel_mask, attention_mask, token_type_ids) -> logits
    inpts: dict-like
        {
          "input_ids":      (B, L),
          "pixel_values":   (B, C, H, W),
          "pixel_mask":     (B, ...),
          "token_type_ids": (B, L),
          "attention_mask": (B, L),
        }
    target_idx: Tensor
        (B,) ground-truth class indices.
    pixel_attribution: Tensor
        (B, H, W) or (B, 1, H, W) or (B, C, H, W): saliency over image.
    perturbation_steps: list/tuple of float
        Fractions of pixels to perturb, e.g. [0.1, 0.2, ..., 0.9].
    mask_value: float
        Value to put in perturbed pixels.
    descending: bool
        If True, remove most-attributed pixels first (deletion curve).
        If False, remove least-attributed pixels first (negative curve).

    Returns
    -------
    dict with keys (all np.ndarray):
        - "baseline_correct":         (B,)
        - "baseline_dissimilarity":   (B,)
        - "prob_diff":                (S, B)
        - "logit_diff":               (S, B)
        - "correct_perturb":          (S, B)
        - "dissimilarity_perturb":    (S, B)
    """
    device = model_wrapper.device
    pixel_values = pixel_values.to(device)
    pixel_attribution = pixel_attribution.to(device)
    target_idx = target_idx.to(device).long()

    # ---------- normalize shapes & define feature/position dims ----------
    ndim = pixel_values.ndim
    origin_shape = pixel_values.shape

    if ndim == 4: # (B, C, H, W)
        B, C, H, W = origin_shape
        num_pixels = H * W
        feat = pixel_values.view(B, C, num_pixels) # (B, C, H*W)
    elif ndim == 3: # (B, num_patches, patch_dim)
        B, num_pixels, patch_dim = origin_shape
        feat = pixel_values.transpose(1, 2) # (B, patch_dim, num_patches)
    else:
        raise ValueError("The pixel values do not have the good dim")

    if pixel_attribution.ndim == 3: # (B, H, W)
        sal_flat = pixel_attribution.view(B, -1) # (B, num_pixels)
    elif pixel_attribution.ndim == 2: # (B, num_patches)
        sal_flat = pixel_attribution
    else:
        raise ValueError("pixel_attribution must be (B, H, W) or (B,num_patches).")
    
    # print(f"feat size {feat.shape} - {pixel_values.shape}")
    # print(f"sal size{sal_flat.shape} - {pixel_attribution.shape}")

    # ---- baseline forward ----
    baseline_logits = forward_fn(pixel_values)        # (B, C)
    if use_sigmoid:
        baseline_probs = baseline_logits.sigmoid()           # (B, C)
    else:
        baseline_probs = baseline_logits.log_softmax(dim=-1)           # (B, C)

    baseline_max_logit, baseline_max_idx = baseline_logits.max(dim=1)   # (B,), (B,)
    baseline_top_scores, baseline_top_ids = baseline_probs.topk(2, dim=1)  # (B,2),(B,2)

    baseline_dissimilarity_t = torch.log(
        baseline_top_scores[:, 0] / baseline_top_scores[:, 1]
    )                                                   # (B,)
    baseline_correct_t = (baseline_top_ids[:, 0] == target_idx)  # (B,)

    baseline_dissimilarity = baseline_dissimilarity_t.float().cpu().numpy()      # (B,)
    baseline_correct = baseline_correct_t.cpu().numpy()                  # (B,)

    # ---- allocate perturbation metrics ----
    S = len(perturbation_steps)
    prob_diff = np.zeros((S, B), dtype=np.float32)
    logit_diff = np.zeros((S, B), dtype=np.float32)
    correct_perturb = np.zeros((S, B), dtype=np.float32)
    dissimilarity_perturb = np.zeros((S, B), dtype=np.float32)
    target_logit_pert = np.zeros((S, B), dtype=np.float32)
    target_prob_pert = np.zeros((S, B), dtype=np.float32)

    num_feats = feat.shape[1] # C or patch_dim

    # ---- perturbation loop ----
    for i, step in enumerate(perturbation_steps):
        k = int(round(step * num_pixels))
        k = max(1, min(k, num_pixels))

        # top-k spatial indices per image
        _, flat_idx = torch.topk(
            sal_flat,
            k,
            dim=-1,
            largest=descending,
        )  # (B, k)


        # clone pixel values and mask selected pixels across channels
        feat_pert = feat.clone()
        # expand indices to channels
        flat_idx_expanded = flat_idx.unsqueeze(1).expand(B, num_feats, k) # (B, num_feats, k)
        # build src tensor filled with mask_value
        mask_src = torch.full_like(feat_pert[:, :, :k], mask_value)
        # scatter over flattened spatial dimension
        feat_pert.scatter_(
            dim=2,
            index=flat_idx_expanded,
            src=mask_src,
        ) # (B, num_feats, k)

        # Reshape back to original pixel_values
        if ndim == 4:
            B, C, H, W = origin_shape
            perturbed_pixels = feat_pert.view(B, C, H, W)       # (B, C, H, W)
        elif ndim == 3:
            perturbed_pixels = feat_pert.transpose(1, 2)        # (B, P, D)
        else:
            raise ValueError("Wrong dim for the pixel values !")

        # forward with perturbed pixels
        logits_pert = forward_fn(perturbed_pixels)  # (B, C)
        if use_sigmoid:
            probs_pert = logits_pert.sigmoid()                                             # (B, C)
        else:
            probs_pert = logits_pert.log_softmax(dim=-1)                                             # (B, C)

        pert_max_logit, pert_max_idx = logits_pert.max(dim=1)      # (B,), (B,)
        pert_top_scores, _ = probs_pert.topk(2, dim=1)             # (B, 2)

        # probability delta (top-1 prob)
        prob_delta = pert_top_scores[:, 0] - baseline_top_scores[:, 0]  # (B,)
        prob_diff[i] = prob_delta.float().cpu().numpy()

        # logit delta (top-1 logit)
        logit_delta = pert_max_logit - baseline_max_logit              # (B,)
        logit_diff[i] = logit_delta.float().cpu().numpy()

        # dissimilarity after perturbation: log(p_target / p_second)
        pert_target_prob = probs_pert[torch.arange(B, device=device), target_idx]  # (B,)
        pert_second_prob = pert_top_scores[:, 1]                                   # (B,)
        pert_dissimilarity_t = torch.log(pert_target_prob / pert_second_prob)      # (B,)
        dissimilarity_perturb[i] = pert_dissimilarity_t.float().cpu().numpy()
        target_prob_pert[i] = pert_target_prob.float().cpu().numpy()
        target_logit_pert[i] = pert_max_logit.float().cpu().numpy()

        # accuracy after perturbation: top-1 vs target
        pert_correct_t = (pert_max_idx == target_idx)                     # (B,)
        correct_perturb[i] = pert_correct_t.float().cpu().numpy()

    return {
        "baseline_correct": baseline_correct,               # (B,)
        "baseline_dissimilarity": baseline_dissimilarity,   # (B,)
        "target_prob_pert": target_prob_pert,               # (S, B)
        "target_logit_pert": target_logit_pert,             # (S, B)
        "prob_diff": prob_diff,                             # (S, B)
        "logit_diff": logit_diff,                           # (S, B)
        "correct_perturb": correct_perturb,                 # (S, B)
        "dissimilarity_perturb": dissimilarity_perturb,     # (S, B)
    }

@torch.no_grad()
def eval_token_perturbation_batch(
    model_wrapper,
    # inpts,
    input_ids: Tensor,
    attention_mask: Tensor,
    target_idx: Tensor,          # (B,) tensor of class indices
    token_attribution: Tensor,   # (B, L) token-wise attribution scores
    perturbation_steps: Sequence[float],
    forward_fn: Callable[[Tensor, Tensor], Tensor],
    mask_token_id: int,
    pad_token_id: int,
    descending: bool = True,
    use_sigmoid: bool = False, # Use sigmoid to compute the probs - False use log_softmax,
    text_token_mask: Optional[Tensor] = None, # Bool of size (B, L) or None
) -> Dict[str, np.ndarray]:
    """
    Batch-level token perturbation evaluation for a VLM wrapper.

    Args
    ----
    model_wrapper: nn.Module
        Your wrapper with a forward like:
            model_wrapper(text_embeds, pixel_values, pixel_mask, attention_mask, token_type_ids)
        and a `.text_embeddings_layer(input_ids)` method.
    inpts: dict-like
        {
          "input_ids":      (B, L),
          "pixel_values":   (B, ...),
          "pixel_mask":     (B, ...),
          "token_type_ids": (B, L),
          "attention_mask": (B, L),
        }
    target_idx: Tensor
        (B,) ground-truth class indices for each sample.
    token_attribution: Tensor
        (B, L) attribution scores per token.
    perturbation_steps: list/tuple of float
        Fractions of tokens to perturb, e.g. [0.1, 0.2, ..., 0.9].
    mask_token_id: int
        Token id that will replace perturbed tokens.
    pad_token_id: int
        Padding token id. If mask_token_id == pad_token_id,
        the attention mask is updated accordingly.
    descending: bool
        If True, mask most important tokens first (positive perturbation).
        If False, mask least important tokens first (negative perturbation).

    Returns
    -------
    dict with keys (all np.ndarray):
        - "baseline_correct":         (B,)
        - "baseline_dissimilarity":   (B,)
        - "prob_diff":                (S, B)
        - "logit_diff":               (S, B)
        - "correct_perturb":          (S, B)
        - "dissimilarity_perturb":    (S, B)
    """
    device = model_wrapper.device
    target_idx = target_idx.to(device)                  # (B,)
    token_attribution = token_attribution.to(device)

    B, L = input_ids.shape

    # ---- baseline forward ----
    baseline_logits = forward_fn(input_ids, attention_mask)       # (B, C)
    if use_sigmoid:
        baseline_probs = baseline_logits.sigmoid()          # (B, C)  # multi-label-ish
    else:
        baseline_probs = baseline_logits.log_softmax(dim=-1)

    baseline_max_logit, baseline_max_idx = baseline_logits.max(dim=1)  # (B,), (B,)
    baseline_top_scores, baseline_top_ids = baseline_probs.topk(2, dim=1)  # (B, 2), (B, 2)

    # per-sample dissimilarity: log(p_top1 / p_top2)
    baseline_dissimilarity_t = torch.log(
        baseline_top_scores[:, 0] / baseline_top_scores[:, 1]
    )  # (B,)

    # correctness: is top-1 predicted equal to target?
    baseline_correct_t = (baseline_top_ids[:, 0] == target_idx).float()  # (B,)

    # store as numpy
    baseline_dissimilarity = baseline_dissimilarity_t.float().cpu().numpy()      # (B,)
    baseline_correct = baseline_correct_t.float().cpu().numpy()                  # (B,)

    # ---- allocate perturbation metrics ----
    S = len(perturbation_steps)
    prob_diff = np.zeros((S, B), dtype=np.float32)
    logit_diff = np.zeros((S, B), dtype=np.float32)
    correct_perturb = np.zeros((S, B), dtype=np.float32)
    dissimilarity_perturb = np.zeros((S, B), dtype=np.float32)
    target_logit_pert = np.zeros((S, B), dtype=np.float32)
    target_prob_pert = np.zeros((S, B), dtype=np.float32)

    # ---- text mask ----
    if text_token_mask is not None:
        text_token_mask = text_token_mask.to(device).bool()  # (B, L)
    else:
        # by default: all positions are perturbable
        text_token_mask = torch.ones(B, L, dtype=torch.bool, device=device)

    # ---- perturbation loop ----
    for i, step in enumerate(perturbation_steps):
        # make a *per-sample* selection mask (B, L)
        select_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        for b in range(B):
            # attribution and mask for that sample
            attr_b = token_attribution[b].clone()         # (L,)
            mask_b = text_token_mask[b]                   # (L,)

            # only consider text tokens
            valid_idx = mask_b.nonzero(as_tuple=False).view(-1)   # (M,)
            if valid_idx.numel() == 0:
                continue  # nothing to perturb in this sample

            # how many text tokens to perturb for this step?
            M = valid_idx.numel()
            k_b = int(round(step * M))
            k_b = max(1, min(k_b, M))

            # restrict attribution to valid positions
            attr_valid = attr_b[valid_idx]               # (M,)
            topk_rel = torch.topk(attr_valid, k_b, largest=descending)
            topk_idx_valid = topk_rel.indices           # (k_b,)
            topk_idx = valid_idx[topk_idx_valid]        # (k_b,) indices in [0, L)

            select_mask[b, topk_idx] = True
    

        # create perturbed input ids: (B, L)
        perturbed_ids = input_ids.clone()
        perturbed_ids[select_mask] = mask_token_id

        # adjust attention_mask only if we truly pad
        perturbed_mask = attention_mask.clone()
        if mask_token_id == pad_token_id:
            # perturbed_mask[batch_idx, indices] = 0
            perturbed_mask[select_mask] = 0


        # ---- perturbed forward ----
        logits_pert = forward_fn(perturbed_ids, perturbed_mask)    # (B, C)
        if use_sigmoid:
            probs_pert = logits_pert.sigmoid()                # (B, C)
        else:
            probs_pert = logits_pert.log_softmax(dim=-1)

        pert_max_logit, pert_max_idx = logits_pert.max(dim=1)     # (B,), (B,)
        pert_top_scores, _ = probs_pert.topk(2, dim=1)            # (B, 2)

        # Probability delta (top-1 prob change)
        prob_delta = pert_top_scores[:, 0] - baseline_top_scores[:, 0]   # (B,)
        prob_diff[i] = prob_delta.float().cpu().numpy()

        # Logit delta (top-1 logit change)
        logit_delta = pert_max_logit - baseline_max_logit                # (B,)
        logit_diff[i] = logit_delta.float().cpu().numpy()

        # Confidence margin after perturbation: log(p_target / p_second)
        pert_target_prob = probs_pert[torch.arange(B, device=device), target_idx]  # (B,)
        pert_second_prob = pert_top_scores[:, 1]                                   # (B,)
        pert_dissimilarity_t = torch.log(pert_target_prob / pert_second_prob)      # (B,)
        dissimilarity_perturb[i] = pert_dissimilarity_t.float().cpu().numpy()
        target_logit_pert[i] = pert_max_logit.float().cpu().numpy()
        target_prob_pert[i] = pert_target_prob.float().cpu().numpy()
        

        # Accuracy after perturbation: top-1 vs target
        pert_correct_t = (pert_max_idx == target_idx).float()                      # (B,)
        correct_perturb[i] = pert_correct_t.float().cpu().numpy()

    return {
        "baseline_correct": baseline_correct,               # (B,)
        "baseline_dissimilarity": baseline_dissimilarity,   # (B,)
        "target_prob_pert": target_prob_pert,               # (S, B)
        "target_logit_pert": target_logit_pert,             # (S, B)
        "prob_diff": prob_diff,                             # (S, B)
        "logit_diff": logit_diff,                           # (S, B)
        "correct_perturb": correct_perturb,                 # (S, B)
        "dissimilarity_perturb": dissimilarity_perturb,     # (S, B)
    }

def compute_auc(pos_pert_logits, neg_pert_logits, perturbation_steps):
    f_pos = pos_pert_logits.mean(axis=-1)
    f_neg = neg_pert_logits.mean(axis=-1)

    auc_pos = float(np.trapezoid(f_pos, perturbation_steps))
    auc_neg = float(np.trapezoid(f_neg, perturbation_steps))
    diff = f_neg - f_pos
    auc_gap = float(np.trapezoid(diff, perturbation_steps))
    return {
        'positive_auc'  : auc_pos,
        'negative_auc'  : auc_neg,
        'gap_auc'       : auc_gap,
    }
