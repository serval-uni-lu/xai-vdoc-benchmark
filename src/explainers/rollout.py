import contextlib
from typing import Tuple, Optional, List
import torch
import torch.nn.functional as F
from src.models import BaseVLMWrapper
from src.explainers import BaseExplainer
from src.explainers.utils import align_llm_visuals_to_pixels

def stitch_chunks_to_matrix(chunk_list):
    """Helper to stitch block-diagonal chunks for Qwen-VL."""
    if not isinstance(chunk_list, list) or len(chunk_list) == 1:
        return chunk_list[0] if isinstance(chunk_list, list) else chunk_list
        
    batch_size, num_heads, _, _ = chunk_list[0].shape
    total_seq_len = sum(chunk.shape[2] for chunk in chunk_list)
    full_matrix = torch.zeros((batch_size, num_heads, total_seq_len, total_seq_len), 
                              device=chunk_list[0].device, dtype=chunk_list[0].dtype)
    current_idx = 0
    for chunk in chunk_list:
        chunk_len = chunk.shape[2]
        full_matrix[:, :, current_idx:current_idx+chunk_len, current_idx:current_idx+chunk_len] = chunk
        current_idx += chunk_len
    return full_matrix

class RolloutExplainer(BaseExplainer):
    def __init__(self, model_wrapper: BaseVLMWrapper, requires_grad=False):
        super().__init__(model_wrapper)
        self.model = self.wrapper.model
        self.requires_grad = requires_grad  # Set to True for Grad Rollout
        
        self.vision_attentions = []
        self.llm_attentions = []
        self.vision_grads = []
        self.llm_grads = []
        self.hooks = []

    def _generic_hook(self, module, output, attns_list, grads_list):
        """A single, smart hook to handle both Vision and LLM safely."""
        # 1. Safely extract the weights (Stashed OR from output tuple)
        attn_data = getattr(module, "saved_attn_weights", None)
        if attn_data is None:
            # Fallback to standard Hugging Face tuple output
            attn_data = output[-1] 
            
        # 2. Force it into a list so we can handle standard models and Qwen interchangeably
        chunk_list = attn_data if isinstance(attn_data, list) else [attn_data]

        if self.requires_grad:
            attns_list.append(chunk_list)
            layer_grads = [None] * len(chunk_list)
            
            def create_hook(idx, target_list):
                return lambda grad: target_list.__setitem__(idx, grad)

            for i, chunk in enumerate(chunk_list):
                if chunk.requires_grad:
                    chunk.retain_grad()
                    chunk.register_hook(create_hook(i, layer_grads))
            grads_list.append(layer_grads)
        else:
            # Detach and CPU offload to save VRAM
            cpu_chunks = [chunk.detach().cpu() for chunk in chunk_list]
            attns_list.append(cpu_chunks)

        # Cleanup stash to prevent memory leaks
        module.saved_attn_weights = None

    def _vision_hook(self, module, input, output):
        self._generic_hook(module, output, self.vision_attentions, self.vision_grads)

    def _llm_hook(self, module, input, output):
        self._generic_hook(module, output, self.llm_attentions, self.llm_grads)


    def register_hooks(self):
        self.clear_hooks()
        for _, module in self.model.named_modules():
            if self.wrapper.vision_module_name and self.wrapper.vision_module_name in str(type(module)):
                self.hooks.append(module.register_forward_hook(self._vision_hook))
            elif self.wrapper.llm_module_name and self.wrapper.llm_module_name in str(type(module)):
                self.hooks.append(module.register_forward_hook(self._llm_hook))
                
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.vision_attentions.clear()
        self.llm_attentions.clear()
        self.vision_grads.clear()
        self.llm_grads.clear()
        torch.cuda.empty_cache()
    
    @contextlib.contextmanager
    def manage_explainability_state(self):
        """
        Temporarily patches the model and attaches hooks. 
        Guarantees complete restoration upon exit.
        """
        # Save the original blueprint and apply the patch
        self.wrapper.apply_patch()
        
        # Register all PyTorch hooks
        self.register_hooks()
        
        try:
            # Yield control back to the main function
            yield 
        finally:
            # This runs NO MATTER WHAT (even if your math throws an error)
            self.clear_hooks()
            
            # Restore the original Hugging Face blueprint
            self.wrapper.remove_patch()

    
    def _compute_rollout_math(self, attentions, gradients=None) -> torch.Tensor:
        """The core mathematical engine for Rollout."""
        if not attentions:
            raise ValueError("Need attention weights to compute Attention (Grad)Rollout")

        # Stitch chunks if dealing with Qwen Vision
        stitched_attns = [stitch_chunks_to_matrix(layer) for layer in attentions]
        
        seq_len = stitched_attns[0].size(-1)
        rollout = torch.eye(seq_len).to(stitched_attns[0].device)

        if gradients is not None and len(gradients) > 0:
            # GRADIENT ROLLOUT
            stitched_grads = [stitch_chunks_to_matrix(layer) for layer in gradients]
            # Gradients populate backwards, so we reverse them to align with attentions
            stitched_grads.reverse() 

            for attn, grad in zip(stitched_attns, stitched_grads):
                grad = torch.clamp(grad, min=0.0) # ReLU
                grad_attn = (attn * grad).mean(dim=1).squeeze(0)
                grad_attn = grad_attn + torch.eye(seq_len).to(grad_attn.device)
                grad_attn = F.normalize(grad_attn, p=1, dim=-1)
                rollout = torch.matmul(rollout, grad_attn)
        else:
            # STANDARD ROLLOUT
            for attn in stitched_attns:
                attn_fused = attn.mean(dim=1).squeeze(0)
                attn_fused = attn_fused + torch.eye(seq_len).to(attn_fused.device)
                attn_fused = F.normalize(attn_fused, p=1, dim=-1)
                rollout = torch.matmul(rollout, attn_fused)

        return rollout
    
    def get_raw_attributions(self, image,
                            text,
                            target_indices: Optional[int | List[int]],
                            average=False,
                            **kwargs
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the model, extracts attentions, and returns Vision and LLM rollouts."""
        
        inputs = self.wrapper.get_inputs(image, text)

        full_ids = kwargs.get("full_ids", None) # (seq_len,)
        if full_ids is None:
            raise ValueError("You should pass the generated_ids tensor")
        
        # Define the indices of the answers tokens and visual tokens 
        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]

        inputs["input_ids"] = full_ids.clone().unsqueeze(0)  # (batch, seq_len)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        with self.manage_explainability_state():    

            if self.requires_grad: # GradientxRollout
                outputs = self.model(**inputs)
                logits = outputs.logits[:, t_start-1:t_end-1, :] # (1, num_ans_tokens, vocab_size)

                new_ids = full_ids[t_start:t_end] #.unsqueeze(0).unsqueeze(-1) # (1, num_ans_tokens, 1)
                new_ids = new_ids.unsqueeze(0).unsqueeze(-1)

                target_logits = logits.gather(dim=-1, index=new_ids).squeeze(-1) # (1, num_ans_tokens)
                answer_score = target_logits.sum()

                self.model.zero_grad()
                answer_score.backward(retain_graph=True)
                
                v_rollout = self._compute_rollout_math(self.vision_attentions,
                                                       self.vision_grads)
                t_rollout = self._compute_rollout_math(self.llm_attentions,
                                                       self.llm_grads)
                v_rollout = v_rollout.detach().cpu()
                t_rollout = t_rollout.detach().cpu()
            else: # Standard Rollout
                with torch.no_grad():
                    _ = self.model(**inputs)
                v_rollout = self._compute_rollout_math(self.vision_attentions)
                t_rollout = self._compute_rollout_math(self.llm_attentions)


        ## Compute the final attribution

        # Token attribution        
        # Create Modality Masks
        image_token_id = self.model.config.image_token_id
        is_image_mask = (full_ids == image_token_id)
        is_image_mask = is_image_mask.cpu()
        is_text_mask = ~is_image_mask

        prompt_mask = torch.arange(full_ids.size(-1),) < t_start
        final_text_mask = is_text_mask & prompt_mask
        final_image_mask = is_image_mask & prompt_mask


        token_attribution = t_rollout[t_start:t_end, final_text_mask]

        if average:
            # Average across the generated sentence to get one score per prompt word
            # Shape: [Num_Prompt_Tokens]
            token_attribution = token_attribution.mean(dim=0)

        # Image attribution
        # Slice the LLM attention (Shape: [Num_Generated_Tokens, Num_Image_Tokens])
        text_to_image_attn = t_rollout[t_start:t_end, final_image_mask]
    

        text_to_vit_attn = align_llm_visuals_to_pixels(text_to_image_attn, inputs)

        # if pixel_values.ndim >= 3 and pixel_values.shape[-3] in [1, 3, 4]:
        #     # Standard VLM and shape is (C, H, W) or (batch_size, C, H, W)
        #     text_to_vit_attn = text_to_image_attn
        #     num_pixels = v_rollout.shape[0]
        
        # else:
        #     # Shape is (grid_h*grid_w, patch_dim)

        #     _, grid_h, grid_w = inputs['image_grid_thw'][0].cpu().numpy().tolist()
        #     spatial_merge_size = 2
        #     llm_grid_h = grid_h // spatial_merge_size
        #     llm_grid_w = grid_w // spatial_merge_size
        #     num_vit_tokens = grid_h * grid_w

        #     # Reshape LLM attention to its 2D grid
        #     # Shape: [Num_Generated_Tokens, 1, grid_h, grid_w]
        #     llm_attn_2d = text_to_image_attn.reshape(-1, 1, llm_grid_h, llm_grid_w)

        #     # Spatially upsample the grid to match the ViT's resolution!
        #     # We use 'nearest' so the LLM's attention is evenly distributed to the 4 sub-patches
        #     vit_attn_2d = torch.nn.functional.interpolate(
        #         llm_attn_2d, 
        #         scale_factor=spatial_merge_size, 
        #         mode='nearest'
        #     )

        #     # Flatten back to 1D to match the ViT Rollout
        #     text_to_vit_attn = vit_attn_2d.reshape(-1, num_vit_tokens)

        # Fuse informations from visual matrices and visual_tokens matrices
        # [gen_len, num_pixels] x [num_pixels, num_pixels] -> [gen_len, num_pixels]
        pixel_attribution = torch.matmul(text_to_vit_attn, v_rollout)

        
        if average:
            pixel_attribution = pixel_attribution.mean(dim=0)


        return token_attribution, pixel_attribution

