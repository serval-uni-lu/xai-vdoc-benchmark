import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, List

from src.models import BaseVLMWrapper


def score_output(model: BaseVLMWrapper,
                inputs: Dict[str, Any],
                input_ids: torch.Tensor,
                pixel_values: torch.Tensor,
                output_ids: torch.Tensor,
                positions: List[torch.Tensor],
                ) -> torch.Tensor:
    generated_ids = torch.cat((input_ids, output_ids), dim=1)

    probs_pred = pred_probs(model,
                            inputs,
                            generated_ids,
                            pixel_values,
                            output_ids,
                            )
    # Fix
    batch_size = probs_pred.shape[0]
    device = probs_pred.device
    
    # FIX: Vectorized position scoring
    # Pre-allocate the result tensor on the GPU
    batch_scores = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        p = positions[b].to(device)
        if len(p) > 0:
            # Gather and mean directly on the GPU, much faster than appending to a list
            # batch_scores[b] = probs_pred[b, p].sum()
            # Normalize by length to get the average log probability per token
            log_prob_sum = probs_pred[b, p].sum() / len(p)

            # 2. Convert back to raw probability space [0, 1] for Game Theory Math!
            batch_scores[b] = torch.exp(log_prob_sum)
            
    scores = batch_scores.cpu()

    return scores

def pred_probs(model: BaseVLMWrapper,
               inputs: Dict[str, Any],
               new_input_ids: torch.Tensor,
               pixel_values: Optional[torch.Tensor],
               output_ids: torch.Tensor,
               ) -> torch.Tensor:

    device = model.device
    
    # attention_mask = torch.ones_like(new_input_ids).to(device)
    pad_token_id = model.processor.tokenizer.pad_token_id if model.processor.tokenizer.pad_token_id is not None else 0
    attention_mask = (new_input_ids != pad_token_id).long().to(device)

    other_kwargs = {k: v \
                    for k, v in inputs.items() \
                    if k not in ['input_ids','pixel_values', 'attention_mask']}
    
    # Intercept and flatten 5D tensors for InternVL/AnyRes models
    if pixel_values is not None and pixel_values.ndim == 5:
        B, num_tiles, C, H, W = pixel_values.shape
        # Flatten (B, num_tiles) into a single batch dimension for the Vision Tower
        pixel_values = pixel_values.view(B * num_tiles, C, H, W)

    with torch.no_grad():
        outputs = model.model(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            #return_probs=False,
            **other_kwargs,
        ) # [batch_size, seq_len, vocab_size]
    all_logits = outputs.logits

    returned_logits = all_logits[:, -output_ids.shape[-1]-1:-1, :] # The reason for the minus 1 is that the generated content is in the previous position
    # returned_logits = F.softmax(returned_logits, dim=-1) # (batch_size, selected_tokens, vocab_size)
    returned_logits = F.log_softmax(returned_logits, dim=-1) # (batch_size, selected_tokens, vocab_size)
    

    returned_logits = returned_logits.gather(dim=2, index=output_ids.unsqueeze(-1)).squeeze(-1) # (batch_size, selected_tokens)
    return returned_logits

def get_most_important_tokens_pixel(model: BaseVLMWrapper,
                                    inputs: Dict[str, Any],
                                    input_ids: torch.Tensor,
                                    target_ids: torch.Tensor,
                                    pixel_values: torch.Tensor,
                                    blur_baseline: torch.Tensor,
                                    # device: torch.device,
                                    ) -> List[torch.Tensor]:
    # Run keyword filtering logic (Batched simplified version of find_keywords)
    # We need probabilities for Original vs Blur
    target_positions = []
    B = input_ids.shape[0]
    with torch.no_grad():
        # Combine prompt + answer
        full_input_ids = torch.cat((input_ids, target_ids), dim=1)
        
        # Forward Original
        # out_orig = model(pixel_values=pixel_values, input_ids=full_input_ids)
        # logits_orig = out_orig.logits[:, input_ids.shape[1]-1 : -1, :] # Shift for next-token prediction
        # probs_orig = torch.gather(logits_orig.softmax(-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
        probs_orig = pred_probs(model=model,
                                inputs=inputs,
                                new_input_ids=full_input_ids,
                                pixel_values=pixel_values,
                                output_ids=target_ids,
                                )

        # Forward Blur
        # out_blur = model(pixel_values=blur_baseline.to(device), input_ids=full_input_ids)
        # logits_blur = out_blur.logits[:, input_ids.shape[1]-1 : -1, :]
        # probs_blur = torch.gather(logits_blur.softmax(-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
        probs_blur = pred_probs(model=model,
                                inputs=inputs,
                                new_input_ids=full_input_ids,
                                pixel_values=blur_baseline,
                                output_ids=target_ids,
                                )

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
    return target_positions

def get_most_important_tokens_token(model: BaseVLMWrapper,
                                   inputs: Dict[str, Any],
                                   input_ids: torch.Tensor,
                                   baseline_input_ids: torch.Tensor,
                                   target_ids: torch.Tensor,
                                   ) -> List[torch.Tensor]:
    # Adapted logic for text
    pixel_values = inputs.get('pixel_values', None)
    
    with torch.no_grad():
        # Score Original
        full_orig = torch.cat((input_ids, target_ids), dim=1)
        probs_orig = pred_probs(model=model,
                                inputs=inputs,
                                new_input_ids=full_orig,
                                pixel_values=pixel_values,
                                output_ids=target_ids)
        
        # Score Baseline (Empty Prompt)
        full_base = torch.cat((baseline_input_ids, target_ids), dim=1)
        probs_base = pred_probs(model=model,
                                inputs=inputs,
                                new_input_ids=full_base,
                                pixel_values=pixel_values,
                                output_ids=target_ids)
        
        # Calculate Ratio
        target_positions = []
        B = input_ids.shape[0]
        ratio_mask = (torch.log(probs_orig + 1e-9) - torch.log(probs_base + 1e-9)) > 1.0
        
        for b in range(B):
            valid_indices = torch.where(ratio_mask[b])[0]
            if len(valid_indices) == 0:
                valid_indices = torch.argmax(probs_orig[b] - probs_base[b]).unsqueeze(0)
            target_positions.append(valid_indices)
            
    return target_positions

def get_most_important_tokens_multimodal(
        model: BaseVLMWrapper,
        inputs: Dict[str, Any],
        input_ids: torch.Tensor,
        base_ids: torch.Tensor,
        pixel_val: torch.Tensor,
        base_pix: torch.Tensor,
        target_ids: torch.Tensor
        ) -> List[torch.Tensor]:
    # Determine keywords based on Joint Original vs Joint Baseline
    with torch.no_grad():
        full_orig = torch.cat((input_ids, target_ids), dim=1)
        full_base = torch.cat((base_ids, target_ids), dim=1)
        
        # Joint Original
        probs_orig = pred_probs(model=model,
                                inputs=inputs,
                                new_input_ids=full_orig,
                                pixel_values=pixel_val,
                                output_ids=target_ids)
        # Joint Baseline (Blur + Pad)
        probs_base = pred_probs(model=model,
                                inputs=inputs,
                                new_input_ids=full_base,
                                pixel_values=base_pix,
                                output_ids=target_ids)
        
        # Calculate Ratio
        target_positions = []
        B = input_ids.shape[0]
        ratio_mask = (torch.log(probs_orig + 1e-9) - torch.log(probs_base + 1e-9)) > 1.0
        
        for b in range(B):
            valid_indices = torch.where(ratio_mask[b])[0]
            if len(valid_indices) == 0:
                valid_indices = torch.argmax(probs_orig[b] - probs_base[b]).unsqueeze(0)
            target_positions.append(valid_indices)
            
    return target_positions
