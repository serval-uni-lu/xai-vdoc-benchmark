import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Sequence, Dict, Optional, Callable, Any


@torch.no_grad()
def eval_image_perturbation_batch(
    model,
    inputs,
    pixel_values: Tensor,           # (B, C, H, W)
    input_ids: Tensor,              # (B, L_prompt)
    target_ids: Tensor,             # (B, L_label) - The expected answer tokens
    pixel_attribution: Tensor,      # (B, H, W)
    perturbation_steps: Sequence[float],
    mask_value: float = 0.0,
    descending: bool = True,        # True = Deletion (remove important first), False = Insertion
    filter_keywords: bool = True,   # If True, only tracks tokens that are "visually dependent"
    blur_baseline: Optional[Tensor] = None # Optional blurred image for keyword filtering
) -> Dict[str, np.ndarray]:
    """
    Batch-level image perturbation evaluation specifically for VLMs.
    
    Adapts the logic of 'metric()' and 'score_output()' into a batched efficient format.
    """
    device = model.device
    pixel_values = pixel_values.to(device)
    pixel_attribution = pixel_attribution.to(device)
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    B, C, H, W = pixel_values.shape
    num_pixels = H * W
    
    # 1. Prepare Target Positions (The "visual keywords" logic)
    # We need to decide WHICH tokens in the target_ids we care about.
    # Default: Track all tokens in the target.
    # Smart: Track only tokens that drop in probability when image is blurred (find_keywords logic).
    
    target_positions = [] # List of tensors, one per batch item
    
    if filter_keywords and blur_baseline is not None:
        # Run keyword filtering logic (Batched simplified version of find_keywords)
        # We need probabilities for Original vs Blur
        with torch.no_grad():
            # Combine prompt + answer
            full_input_ids = torch.cat((input_ids, target_ids), dim=1)
            
            # Forward Original
            out_orig = model(pixel_values=pixel_values, input_ids=full_input_ids)
            logits_orig = out_orig.logits[:, input_ids.shape[1]-1 : -1, :] # Shift for next-token prediction
            probs_orig = torch.gather(logits_orig.softmax(-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
            
            # Forward Blur
            out_blur = model(pixel_values=blur_baseline.to(device), input_ids=full_input_ids)
            logits_blur = out_blur.logits[:, input_ids.shape[1]-1 : -1, :]
            probs_blur = torch.gather(logits_blur.softmax(-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)

            # Ratio Check: Log(P_orig / P_blur) > 1.0
            # Corresponds to: condition = (torch.log(probs)-torch.log(probs_blur) > 1.0)
            ratio_mask = (torch.log(probs_orig + 1e-9) - torch.log(probs_blur + 1e-9)) > 1.0
            
            for b in range(B):
                # Get indices where condition is true
                valid_indices = torch.where(ratio_mask[b])[0]
                if len(valid_indices) == 0:
                    # Fallback: Pick max drop if no strong keywords found
                    valid_indices = torch.argmax(probs_orig[b] - probs_blur[b]).unsqueeze(0)
                target_positions.append(valid_indices)
    else:
        # Default: Use all target tokens
        seq_len = target_ids.shape[1]
        default_indices = torch.arange(seq_len, device=device)
        target_positions = [default_indices for _ in range(B)]

    # 2. Define Score Function (The "score_output" logic)
    def get_batch_scores(current_images):
        """
        Computes the average log-probability of the target tokens.
        Equivalent to: scores = probs_pred[:, positions].sum(-1) / len(positions)
        """
        full_input_ids = torch.cat((input_ids, target_ids), dim=1)
        outputs = model(pixel_values=current_images, input_ids=full_input_ids)
        
        # Logits for the target sequence
        # We start predicting from the last token of input_ids
        # target_ids[0] is predicted by input_ids[-1]
        shift_logits = outputs.logits[:, input_ids.shape[1]-1 : -1, :]
        shift_labels = target_ids
        
        # Get log probs of the ground truth tokens
        log_probs = shift_logits.log_softmax(dim=-1)
        token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1) # (B, Seq_Len)
        
        # Sum specific positions and normalize
        batch_scores = []
        for b in range(B):
            pos = target_positions[b]
            # Sum log probs of selected keywords
            score_sum = token_log_probs[b, pos].sum()
            # Divide by count (normalization)
            avg_score = score_sum / len(pos)
            batch_scores.append(avg_score)
            
        return torch.stack(batch_scores) # (B,)

    # 3. Setup Baselines & Flattening
    feat = pixel_values.view(B, C, num_pixels) # (B, C, H*W)
    sal_flat = pixel_attribution.view(B, -1)   # (B, H*W)

    # Calculate Baseline Scores
    baselines_logits = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            return_probs=True,
                            **kwargs_dict,
                            )
    baseline_scores = get_batch_scores(pixel_values).cpu().numpy() # (B,)
    
    # If we have a blur baseline, we use its score for normalization (0.0 point)
    # If not, we use the score of a fully masked image (calculated later or assumed)
    if blur_baseline is not None:
        blur_scores = get_batch_scores(blur_baseline.to(device)).cpu().numpy()
    else:
        # Temporary fallback if no blur image provided: assume 0.0 or calculate on fully masked
        blur_scores = np.zeros_like(baseline_scores) # Placeholder

    # 4. Perturbation Loop
    S = len(perturbation_steps)
    scores_perturb = np.zeros((S, B), dtype=np.float32)
    normalized_scores = np.zeros((S, B), dtype=np.float32)

    for i, step in enumerate(perturbation_steps):
        k = int(round(step * num_pixels))
        k = max(1, min(k, num_pixels))

        # Identify pixels to mask
        # Descending = True -> Remove highest saliency first (Deletion)
        _, flat_idx = torch.topk(sal_flat, k, dim=-1, largest=descending) # (B, k)

        # Create Perturbed Batch
        feat_pert = feat.clone()
        flat_idx_expanded = flat_idx.unsqueeze(1).expand(B, C, k)
        mask_src = torch.full_like(feat_pert[:, :, :k], mask_value)
        
        # Apply mask
        feat_pert.scatter_(dim=2, index=flat_idx_expanded, src=mask_src)
        perturbed_images = feat_pert.view(B, C, H, W)

        # Compute Scores
        current_scores = get_batch_scores(perturbed_images).cpu().numpy() # (B,)
        scores_perturb[i] = current_scores

        # Normalize: (Current - Blur) / (Original - Blur)
        # This matches: outputs = (outputs-blur_scores) / (og_scores-blur_scores)
        norm_score = (current_scores - blur_scores) / (baseline_scores - blur_scores + 1e-9)
        normalized_scores[i] = norm_score

    return {
        "baseline_scores": baseline_scores,       # Raw scores of original image
        "blur_scores": blur_scores,               # Raw scores of baseline (blur) image
        "raw_scores_perturb": scores_perturb,     # (S, B) Raw scores at each step
        "normalized_scores": normalized_scores,   # (S, B) Normalized 0-1 scores (AUC ready)
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


    # 2. Define Score Function (The "score_output" logic)
    def get_batch_scores(current_images):
        """
        Computes the average log-probability of the target tokens.
        Equivalent to: scores = probs_pred[:, positions].sum(-1) / len(positions)
        """
        full_input_ids = torch.cat((input_ids, target_ids), dim=1)
        outputs = model(pixel_values=current_images, input_ids=full_input_ids)
        
        # Logits for the target sequence
        # We start predicting from the last token of input_ids
        # target_ids[0] is predicted by input_ids[-1]
        shift_logits = outputs.logits[:, input_ids.shape[1]-1 : -1, :]
        shift_labels = target_ids
        
        # Get log probs of the ground truth tokens
        log_probs = shift_logits.log_softmax(dim=-1)
        token_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1) # (B, Seq_Len)
        
        # Sum specific positions and normalize
        batch_scores = []
        for b in range(B):
            pos = target_positions[b]
            # Sum log probs of selected keywords
            score_sum = token_log_probs[b, pos].sum()
            # Divide by count (normalization)
            avg_score = score_sum / len(pos)
            batch_scores.append(avg_score)
            
        return torch.stack(batch_scores) # (B,)


def score_output(model,
                inputs,
                input_ids,
                pixel_values,
                output_ids,
                positions):
    generated_ids = torch.cat((input_ids, output_ids), dim=1)
    probs_pred = pred_probs(model,
                            inputs,
                            generated_ids,
                            pixel_values,
                            output_ids,
                            ).unsqueeze(0)
    scores = probs_pred[:, torch.tensor(positions).to(probs_pred.device)].sum(-1) 
    return scores / len(positions)

def pred_probs(model,
               inputs: Dict[str, Any],
               new_input_ids: torch.Tensor,
               pixel_values: Optional[torch.Tensor],
               output_ids,
            #    target_token_position,
            #    selected_token_word_id,
               ):
    inputs_new = inputs.copy()
    device = model.device
    
    attention_mask = torch.ones_like(new_input_ids).to(device)   

    with torch.no_grad():
        outputs = model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **inputs_new,
            return_probs=False,
        )
        all_logits = outputs.logits  # [batch_size, seq_len, vocab_size]

    returned_logits = all_logits[:, -output_ids.shape[-1]-1:-1, :] # The reason for the minus 1 is that the generated content is in the previous position
    returned_logits = F.softmax(returned_logits, dim=-1)
    
    # selected_token_word_id = torch.tensor(selected_token_word_id).to(model.device)
    # indices = selected_token_word_id.unsqueeze(0).unsqueeze(-1) # [1, N, 1]
    
    # returned_logits = returned_logits.gather(dim=2, index=indices) # [1, N, 1]
    # returned_logits = returned_logits.squeeze(-1)  # [1, N]
    returned_logits = returned_logits.gather(dim=2, output_ids.unsqueeze(-1)).squeeze()
    
    return returned_logits[0]

def metric(args, image, baseline, mask, model, model_name, label, label_i, pred_data, size=28, prompt=None, image_size=None, positions=None, resolution=None):
    with torch.no_grad():
        # The dimensions for the image
        #img_size = image.shape[-1]
        # Compute the total number of pixels in a mask
        mask_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        num_pixels = torch.prod(torch.tensor(mask.shape[1:])).item()
        # Compute the step size
        step=max(1, num_pixels // 50)
        # Used for indexing with batch sizes
        l = torch.arange(1)
        # The unmasked score
        og_scores = score_output(args, image, image_size, model, model_name, l, label, prompt, positions)
        # The baseline score
        blur_scores = score_output(args, baseline, image_size, model, model_name, l, label, prompt, positions)
        # Initial values for the curves
        del_curve = [og_scores]
        ins_curve = [blur_scores]
        index = [0.]

        # True_mask is used to hold 1 or 0. Either show that pixel or blur it.
        true_mask = torch.ones((mask.shape[0], mask_pixels)).cuda()
        del_scores = torch.zeros(mask.shape[0])
        ins_scores = torch.zeros(mask.shape[0])
        # Sort each mask by values and store the indices.
        elements = torch.argsort(mask.view(mask.shape[0], -1), dim=1)
        for pixels in range(0, num_pixels, step):
            # Get the indices used in this iteration
            indices = elements[l,pixels:pixels+step].squeeze().view(1, -1)
            # Set those indices to 0
            true_mask[l, indices.permute(1,0)] = 0
            up_mask = upscale(true_mask.view(-1, 1, size,size), image, resolution)
            # Mask the image for deletion
            if isinstance(image, list):
                del_image = [phi(x, y, z).half() for x, y, z in zip(image, baseline, up_mask)]
            else:
                del_image = phi(image, baseline, up_mask).half()
            # Calculate new scores
            outputs = score_output(args, del_image, image_size, model, model_name, l, label, prompt, positions)
            del_curve.append(outputs)
            index.append((pixels+step)/num_pixels)
            outputs = (outputs-blur_scores) / (og_scores-blur_scores)
            del_scores += outputs.cpu() * step if pixels + step < num_pixels else num_pixels - pixels

            # Mask the image for insertion
            if isinstance(image, list):
                ins_image = [phi(x, y, z).half() for x, y, z in zip(baseline, image, up_mask)]
            else:
                ins_image = phi(baseline, image, up_mask).half()

            # Calculate the new scores
            outputs = score_output(args, ins_image, image_size, model, model_name, l, label, prompt, positions)

            ins_curve.append(outputs)
            outputs = (outputs-blur_scores) / (og_scores-blur_scores)
            ins_scores += outputs.cpu() * step if pixels + step < num_pixels else num_pixels - pixels

        # Force scores between 0 and 1.
        del_scores /= num_pixels
        ins_scores /= num_pixels

        del_curve = list(map(lambda x: [y.item() for y in x], zip(*del_curve)))
        ins_curve = list(map(lambda x: [y.item() for y in x], zip(*ins_curve)))

    return del_scores, ins_scores, del_curve, ins_curve, index