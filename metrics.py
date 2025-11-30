import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Sequence, Dict, Optional


@torch.no_grad()
def eval_image_perturbation_batch(
    model_wrapper,
    inpts,
    target_idx: Tensor,           # (B,) tensor of class indices
    pixel_attribution: Tensor,    # (B, H, W) or (B, 1, H, W) or (B, C, H, W)
    perturbation_steps: Sequence[float],
    mask_value: float = 0.0,
    descending: bool = True,      # True: most important first; False: least important
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

    input_ids      = inpts["input_ids"].to(device)        # (B, L)
    pixel_values   = inpts["pixel_values"].to(device)     # (B, C, H, W)
    pixel_mask     = inpts["pixel_mask"].to(device)
    token_type_ids = inpts["token_type_ids"].to(device)
    attention_mask = inpts["attention_mask"].to(device)

    target_idx = target_idx.to(device).long()             # (B,)

    B, C, H, W = pixel_values.shape
    text_embeds = model_wrapper.text_embeddings_layer(input_ids)  # (B, L, D)

    # ---- normalize pixel attribution to (B, H*W) ----
    if pixel_attribution.dim() == 3:
        sal = pixel_attribution.to(device)             # (B, H, W)
    else:
        raise ValueError("pixel_attribution must be (B, H, W) or (B, C, H, W)/(B,1,H,W).")

    sal_flat = sal.view(B, -1)  # (B, H*W)

    # ---- baseline forward ----
    captum_forward = (text_embeds, pixel_values)
    captum_add_forward = (pixel_mask, attention_mask, token_type_ids)
    captum_args = captum_forward + captum_add_forward

    baseline_logits = model_wrapper(*captum_args)        # (B, C)
    baseline_probs = baseline_logits.sigmoid()           # (B, C)

    baseline_max_logit, baseline_max_idx = baseline_logits.max(dim=1)   # (B,), (B,)
    baseline_top_scores, baseline_top_ids = baseline_probs.topk(2, dim=1)  # (B,2),(B,2)

    baseline_dissimilarity_t = torch.log(
        baseline_top_scores[:, 0] / baseline_top_scores[:, 1]
    )                                                   # (B,)
    baseline_correct_t = (baseline_top_ids[:, 0] == target_idx).float()  # (B,)

    baseline_dissimilarity = baseline_dissimilarity_t.cpu().numpy()      # (B,)
    baseline_correct = baseline_correct_t.cpu().numpy()                  # (B,)

    # ---- allocate perturbation metrics ----
    S = len(perturbation_steps)
    prob_diff = np.zeros((S, B), dtype=np.float32)
    logit_diff = np.zeros((S, B), dtype=np.float32)
    correct_perturb = np.zeros((S, B), dtype=np.float32)
    dissimilarity_perturb = np.zeros((S, B), dtype=np.float32)
    target_logit_pert = np.zeros((S, B), dtype=np.float32)

    num_pixels = H * W

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
        perturbed_pixels = pixel_values.clone().view(B, C, -1)  # (B, C, H*W)

        # expand indices to channels
        flat_idx_expanded = flat_idx.unsqueeze(1).expand(B, C, k)                # (B,C,k)

        # build src tensor filled with mask_value
        mask_src = torch.full_like(perturbed_pixels[:, :, :k], mask_value)

        # scatter over flattened spatial dimension
        perturbed_pixels.scatter_(
            dim=2,
            index=flat_idx_expanded,
            src=mask_src,
        )
        perturbed_pixels = perturbed_pixels.view(B, C, H, W)

        # forward with perturbed pixels
        captum_forward_pert = (text_embeds, perturbed_pixels)
        captum_add_forward_pert = (pixel_mask, attention_mask, token_type_ids)
        logits_pert = model_wrapper(*(captum_forward_pert + captum_add_forward_pert))  # (B, C)
        probs_pert = logits_pert.sigmoid()                                             # (B, C)

        pert_max_logit, pert_max_idx = logits_pert.max(dim=1)      # (B,), (B,)
        pert_top_scores, _ = probs_pert.topk(2, dim=1)             # (B, 2)

        # probability delta (top-1 prob)
        prob_delta = pert_top_scores[:, 0] - baseline_top_scores[:, 0]  # (B,)
        prob_diff[i] = prob_delta.cpu().numpy()

        # logit delta (top-1 logit)
        logit_delta = pert_max_logit - baseline_max_logit              # (B,)
        logit_diff[i] = logit_delta.cpu().numpy()

        # dissimilarity after perturbation: log(p_target / p_second)
        pert_target_prob = probs_pert[torch.arange(B, device=device), target_idx]  # (B,)
        pert_second_prob = pert_top_scores[:, 1]                                   # (B,)
        pert_dissimilarity_t = torch.log(pert_target_prob / pert_second_prob)      # (B,)
        dissimilarity_perturb[i] = pert_dissimilarity_t.cpu().numpy()
        target_logit_pert[i] = pert_target_prob.cpu().numpy()

        # accuracy after perturbation: top-1 vs target
        pert_correct_t = (pert_max_idx == target_idx).float()                      # (B,)
        correct_perturb[i] = pert_correct_t.cpu().numpy()

    return {
        "baseline_correct": baseline_correct,               # (B,)
        "baseline_dissimilarity": baseline_dissimilarity,   # (B,)
        "target_logit_pert": target_logit_pert,             # (S, B)
        "prob_diff": prob_diff,                             # (S, B)
        "logit_diff": logit_diff,                           # (S, B)
        "correct_perturb": correct_perturb,                 # (S, B)
        "dissimilarity_perturb": dissimilarity_perturb,     # (S, B)
    }



@torch.no_grad()
def eval_token_perturbation_batch(
    model_wrapper,
    inpts,
    target_idx: Tensor,          # (B,) tensor of class indices
    token_attribution: Tensor,   # (B, L) token-wise attribution scores
    perturbation_steps: Sequence[float],
    mask_token_id: int,
    pad_token_id: int,
    descending: bool = True,
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

    input_ids      = inpts["input_ids"].to(device)      # (B, L)
    pixel_values   = inpts["pixel_values"].to(device)
    pixel_mask     = inpts["pixel_mask"].to(device)
    token_type_ids = inpts["token_type_ids"].to(device)
    attention_mask = inpts["attention_mask"].to(device)

    target_idx = target_idx.to(device)                  # (B,)

    B, L = input_ids.shape

    # ---- baseline forward ----
    text_embeds = model_wrapper.text_embeddings_layer(input_ids)  # (B, L, D)

    captum_forward = (text_embeds, pixel_values)
    captum_add_forward = (pixel_mask, attention_mask, token_type_ids)

    captum_args = captum_forward + captum_add_forward
    baseline_logits = model_wrapper(*captum_args)       # (B, C)
    baseline_probs = baseline_logits.sigmoid()          # (B, C)  # multi-label-ish

    baseline_max_logit, baseline_max_idx = baseline_logits.max(dim=1)  # (B,), (B,)
    baseline_top_scores, baseline_top_ids = baseline_probs.topk(2, dim=1)  # (B, 2), (B, 2)

    # per-sample dissimilarity: log(p_top1 / p_top2)
    baseline_dissimilarity_t = torch.log(
        baseline_top_scores[:, 0] / baseline_top_scores[:, 1]
    )  # (B,)

    # correctness: is top-1 predicted equal to target?
    baseline_correct_t = (baseline_top_ids[:, 0] == target_idx).float()  # (B,)

    # store as numpy
    baseline_dissimilarity = baseline_dissimilarity_t.cpu().numpy()      # (B,)
    baseline_correct = baseline_correct_t.cpu().numpy()                  # (B,)

    # ---- allocate perturbation metrics ----
    S = len(perturbation_steps)
    prob_diff = np.zeros((S, B), dtype=np.float32)
    logit_diff = np.zeros((S, B), dtype=np.float32)
    correct_perturb = np.zeros((S, B), dtype=np.float32)
    dissimilarity_perturb = np.zeros((S, B), dtype=np.float32)
    target_logit_pert = np.zeros((S, B), dtype=np.float32)

    # ---- perturbation loop ----
    for i, step in enumerate(perturbation_steps):
        # number of tokens to perturb
        k = int(round(step * L))
        k = max(1, min(k, L))

        # indices of top-k tokens per sample: (B, k)
        _, indices = torch.topk(
            token_attribution.to(device),
            k,
            dim=-1,
            largest=descending,
        )

        # create perturbed input ids: (B, L)
        perturbed_ids = input_ids.clone()

        # build batch index grid for advanced indexing
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)  # (B, k)
        perturbed_ids[batch_idx, indices] = mask_token_id

        # adjust attention_mask only if we truly pad
        perturbed_mask = attention_mask.clone()
        if mask_token_id == pad_token_id:
            perturbed_mask[batch_idx, indices] = 0

        # embeddings from perturbed ids
        perturbed_embeds = model_wrapper.text_embeddings_layer(perturbed_ids)

        captum_forward_pert = (perturbed_embeds, pixel_values)
        captum_add_forward_pert = (pixel_mask, perturbed_mask, token_type_ids)
        captum_args_pert = captum_forward_pert + captum_add_forward_pert

        # ---- perturbed forward ----
        logits_pert = model_wrapper(*captum_args_pert)    # (B, C)
        probs_pert = logits_pert.sigmoid()                # (B, C)

        pert_max_logit, pert_max_idx = logits_pert.max(dim=1)     # (B,), (B,)
        pert_top_scores, _ = probs_pert.topk(2, dim=1)            # (B, 2)

        # Probability delta (top-1 prob change)
        prob_delta = pert_top_scores[:, 0] - baseline_top_scores[:, 0]   # (B,)
        prob_diff[i] = prob_delta.cpu().numpy()

        # Logit delta (top-1 logit change)
        logit_delta = pert_max_logit - baseline_max_logit                # (B,)
        logit_diff[i] = logit_delta.cpu().numpy()

        # Confidence margin after perturbation: log(p_target / p_second)
        pert_target_prob = probs_pert[torch.arange(B, device=device), target_idx]  # (B,)
        pert_second_prob = pert_top_scores[:, 1]                                   # (B,)
        pert_dissimilarity_t = torch.log(pert_target_prob / pert_second_prob)      # (B,)
        dissimilarity_perturb[i] = pert_dissimilarity_t.cpu().numpy()
        target_logit_pert[i] = pert_target_prob.cpu().numpy()
        

        # Accuracy after perturbation: top-1 vs target
        pert_correct_t = (pert_max_idx == target_idx).float()                      # (B,)
        correct_perturb[i] = pert_correct_t.cpu().numpy()

    return {
        "baseline_correct": baseline_correct,               # (B,)
        "baseline_dissimilarity": baseline_dissimilarity,   # (B,)
        "target_logit_pert": target_logit_pert,             # (S, B)
        "prob_diff": prob_diff,                             # (S, B)
        "logit_diff": logit_diff,                           # (S, B)
        "correct_perturb": correct_perturb,                 # (S, B)
        "dissimilarity_perturb": dissimilarity_perturb,     # (S, B)
    }
