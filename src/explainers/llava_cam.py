import contextlib

from typing import Optional, List, Tuple
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

from src.models import BaseVLMWrapper
from src.explainers import BaseExplainer
from src.explainers.utils import align_llm_visuals_to_pixels


class LLaVACAMExplainer(BaseExplainer):
    def __init__(self, 
                 model_wrapper: BaseVLMWrapper,
                 target_layer_name: str,
                 num_samples: int=10,
                 noise_std: float=.1,
                 token_wise = True,
                 ):
        super().__init__(model_wrapper)
        
        self.target_layer_name = target_layer_name
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.token_wise = token_wise


    def save_feature_maps(self, module, input, output):
        """Hook to save the feature maps during forward pass."""
        self.feature_maps = output
        #output.retain_grad()

    def save_gradients(self, module, grad_input, grad_output):
        """Hook to save the gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def register_hooks(self):
        for name, module in self.wrapper.model.named_modules():
            if self.target_layer_name in name:
                module.register_forward_hook(self.save_feature_maps)
                module.register_backward_hook(self.save_gradients)
    
    def clear_hooks(self):
        self.feature_maps = None
        self.gradients = None

        torch.cuda.empty_cache()

    @contextlib.contextmanager
    def manage_explainability_state(self):
        """
        Temporarily patches the model and attaches hooks. 
        Guarantees complete restoration upon exit.
        """
        
        # Register hooks to capture activations (forward) and gradients (backward)
        self.register_hooks()
        
        try:
            # Yield control back to the main function
            yield 
        finally:
            # This runs NO MATTER WHAT (even if your math throws an error)
            self.clear_hooks()
        
    def compute_cam(self, mask):
        """Applies Grad-CAM channel-weighting to the selected tokens."""
        # Slice the sequence down to just the tokens we care about
        # Shape: [num_selected_tokens, Channels]

        if self.gradients is None or self.feature_maps is None:
            raise RuntimeError(f"Gradients or Feature maps were dropped/not initialized")


        activations = self.feature_maps
        gradients = self.gradients
        
        feats = activations[0, mask, :]
        grads = gradients[0, mask, :]

        # Global Average Pooling of the gradients across the tokens
        # Shape: [Channels]
        pooled_grads = grads.mean(dim=0)

        # Weight the feature maps
        # [num_selected_tokens, Channels] * [Channels]
        weighted_feats = feats * pooled_grads

        # Average across the channel dimension to get a single score per token
        # Shape: [num_selected_tokens]
        cam = weighted_feats.sum(dim=-1)

        # Grad-CAM requires a ReLU to keep only positive contributions
        cam = nn.functional.relu(cam)
        
        # # Normalize to [0, 1]
        # if cam.max() > 0:
        #     cam = cam / cam.max()
            
        return cam.detach().cpu()

    def _add_noise(self, image, noise_std):
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.copy().convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        img_arr = np.array(pil_img).astype(np.float32)
        noise = np.random.normal(0, noise_std * 255, img_arr.shape)
        noisy_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)

        noisy_pil = Image.fromarray(noisy_arr)

        return noisy_pil

    
    def get_raw_attributions(self,
                            image, text,
                            target_indices=None,
                            **kwargs
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Grad-CAM
        if self.num_samples <= 1:
            smooth_token_cam, smooth_pixel_cam = self.llava_cam(image=image,
                                                                text=text,
                                                                **kwargs)
            return smooth_token_cam, smooth_pixel_cam

        # SmoothGRAD-CAM (or Llava-CAM)
        # First run to get shape and type
        noisy_img = self._add_noise(image, noise_std=self.noise_std)
        smooth_token_cam, smooth_pixel_cam = self.llava_cam(image=noisy_img, text=text, **kwargs)

        for _ in range(self.num_samples - 1):
            noisy_img = self._add_noise(image, noise_std=self.noise_std)
            t_cam, p_cam = self.llava_cam(image=noisy_img, text=text, **kwargs)
            
            smooth_token_cam += t_cam
            smooth_pixel_cam += p_cam
        
        # Average
        smooth_token_cam /= self.num_samples
        smooth_pixel_cam /= self.num_samples

        return smooth_token_cam.float(), smooth_pixel_cam.float()

    def llava_cam(self,
            image,
            text: str,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates raw attributions. 
        Returns: Tuple of (Attributions, Feature_Maps) of shape (Batch, Seq_Len, Hidden_Dim).
        """

        inputs = self.wrapper.get_inputs(image, text)
        
        pred_results = kwargs.get("pred_results", None)
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs=inputs,
                                            return_logits=False,
                                            **kwargs,
                                            )
        full_ids = pred_results["full_ids"]
        
        # Define the indices of the answers tokens and visual tokens 
        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]
        gen_len = t_end - t_start
                
        inputs["input_ids"] = full_ids.clone().unsqueeze(0)  # (batch, seq_len)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        
        image_token_id = self.wrapper.model.config.image_token_id
        # Boolean Masks (flattened to 1D)
        full_ids_1d = full_ids.squeeze()
        prompt_mask = torch.arange(full_ids_1d.size(-1),
                                   device=full_ids_1d.device) < t_start
        is_image_mask = (full_ids_1d == image_token_id)
        is_text_mask = ~is_image_mask

        final_text_mask = is_text_mask & prompt_mask
        final_image_mask = is_image_mask & prompt_mask

        
        with self.manage_explainability_state():
            outputs = self.wrapper.model(**inputs)
            logits = outputs.logits[:, t_start-1:t_end-1, :] # (1, num_ans_tokens, vocab_size)

            new_ids = full_ids[t_start:t_end] #.unsqueeze(0).unsqueeze(-1) # (1, num_ans_tokens, 1)
            new_ids = new_ids.unsqueeze(0).unsqueeze(-1)
            
            target_logits = logits.gather(dim=-1, index=new_ids).squeeze(-1) # (1, num_ans_tokens)

            if self.token_wise:
                token_attributions = []
                pixel_attributions = []

                # Iterate on each generated token's logit
                for i in range(gen_len):
                    # Backward Pass
                    self.wrapper.model.zero_grad()
                    target_logits[:, i].backward(retain_graph=True)

                    # Safety check: Did the hook catch the gradients?
                    if self.gradients is None:
                        raise RuntimeError(f"Gradients were dropped at iteration {i}")

                    # Compute CAM for each word
                    tok_attr = self.compute_cam(prompt_mask)
                    pix_attr = self.compute_cam(final_image_mask)

                    self.gradients = None

                    token_attributions.append(tok_attr)
                    pixel_attributions.append(pix_attr)
                
                # Stack the list of 1D tensors into a 2D tensor
                token_attribution = torch.stack(token_attributions, dim=0) # [gen_len, num_text_tokens]
                pixel_attribution = torch.stack(pixel_attributions, dim=0) # [gen_len, num_image_tokens]
                

            else:
                # Generate 1 attribution for the whole sequence
                answer_score = target_logits.sum()

                # Backward Pass
                self.wrapper.model.zero_grad()
                answer_score.backward(retain_graph=True)

                # Compute text and image attributions
                token_attribution = self.compute_cam(final_text_mask).unsqueeze(0)
                pixel_attribution = self.compute_cam(final_image_mask).unsqueeze(0)               

        pixel_attribution = align_llm_visuals_to_pixels(pixel_attribution, inputs)

        return token_attribution, pixel_attribution
    
