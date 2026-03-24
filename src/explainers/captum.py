from typing import Literal, Tuple, List, Optional, Dict, Any

import torch
from captum.attr import (
    IntegratedGradients, InputXGradient,
    TokenReferenceBase,
)

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper

class CaptumExplainer(BaseExplainer):
    def __init__(self,
                model_wrapper: BaseVLMWrapper,
                xai_name,
                ):
        super().__init__(model_wrapper)
        self.xai_name = xai_name
        self.explainer = self._get_explainer(self.xai_name)

    def _get_explainer(self,
                       xai_name: Literal["inputxgradient", "integrated"]):
        if xai_name == "inputxgradient":
            xai_method = InputXGradient(self.wrapper.get_captum_forward)
        elif xai_name == "integrated":
            xai_method = IntegratedGradients(self.wrapper.get_captum_forward)
        else:
            raise ValueError(f"Unknow method name {xai_name}")
        return xai_method
    
    def _get_integrated_gradient_kwargs(self) -> Dict[str, Any]:
        integrad_kwargs = {
            "baselines": None,
            "n_steps": 5,
            "internal_batch_size": 1,
        }
        return integrad_kwargs
    
    def get_raw_attributions(self,
                             image,
                             text,
                             target_indices: Optional[int | List[int]],
                             **kwargs,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.wrapper.get_inputs(image, text)

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]  #.unsqueeze(0)

        text_embeds = self.wrapper.embed_text(input_ids)

        captum_forward = (text_embeds.requires_grad_(), pixel_values.requires_grad_())
        kwargs_dict = {k: v for k, v in inputs.items() if k not in ['pixel_values']}
        
        use_baselines = False
        if self.xai_name in ["integrated"]:
            use_baselines = True
       
        pred_results = kwargs.get("pred_results", None)
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs=inputs,
                                            return_logits=False,
                                            **kwargs,
                                            )
        new_ids = pred_results["new_ids"]
        seq_len = len(new_ids)
        # target_token = new_ids[target_indices].cpu().numpy().tolist()

        token_attrs = []
        pixel_attrs = []
        for step_idx in range(seq_len):
            target_token = new_ids[step_idx].item()

            # --- DYNAMIC CONTEXT BUILDING ---
            if step_idx == 0:
                # Predicting the first token: context is just the prompt
                current_input_ids = input_ids 
            else:
                # Predicting token T: context is prompt + previously generated tokens
                step_ids = new_ids[:step_idx].unsqueeze(0) # Shape: (1, step_idx)
                current_input_ids = torch.cat([input_ids, step_ids], dim=1)
            
            # Embed the dynamically growing text sequence
            current_text_embeds = self.wrapper.embed_text(current_input_ids).requires_grad_()
            captum_forward = (current_text_embeds, pixel_values.requires_grad_())

            # Dynamically grow the attention mask if it exists
            step_kwargs = kwargs_dict.copy()
            step_kwargs["input_ids"] = current_input_ids
            if "attention_mask" in step_kwargs:
                # Add 1s for the newly appended tokens
                extra_mask = torch.ones((1, step_idx),
                                        dtype=step_kwargs["attention_mask"].dtype,
                                        device=self.device)
                step_kwargs["attention_mask"] = torch.cat([step_kwargs["attention_mask"],
                                                                extra_mask],
                                                           dim=1)
                
            
            baselines = None
            if use_baselines:
                token_reference = TokenReferenceBase(
                    reference_token_idx=self.wrapper.processor.tokenizer.pad_token_id)
                # generate reference for each sample
                reference_ids = token_reference.generate_reference(
                                        current_input_ids.shape[-1],
                                        device=self.device
                                        ).unsqueeze(0)
                reference_embeds = self.wrapper.embed_text(reference_ids)
                baselines = (reference_embeds, torch.zeros_like(pixel_values))
                

            # Get attributions
            int_kwargs = self._get_integrated_gradient_kwargs()
            int_kwargs["baselines"] = baselines

            if use_baselines:
                token_attr, pixel_attr = self.explainer.attribute(inputs=captum_forward,
                                                        target=target_token,
                                                        additional_forward_args=step_kwargs,
                                                        **int_kwargs
                                                        )
            else:
                token_attr, pixel_attr = self.explainer.attribute(inputs=captum_forward,
                                                    target=target_token,
                                                    additional_forward_args=step_kwargs,
                                                    )
        
            token_attr = token_attr.detach().cpu()
            pixel_attr = pixel_attr.detach().cpu()
            torch.cuda.empty_cache()

            if step_idx > 0 :
                token_attr = token_attr[:, :-step_idx, :] # Remove the attribution for answer tokens
            token_attrs.append(token_attr)
            pixel_attrs.append(pixel_attr)

        token_attrs = torch.cat(token_attrs, dim=0) # (seq_len, input_ids.shape[-1], hidden_dim)
        pixel_attrs = torch.stack(pixel_attrs, dim=0) # (seq_len, num_pixels, hidden_dim)

        token_attrs = token_attrs.sum(-1)

        model_type = getattr(self.wrapper.model.config, "model_type", "").lower()

        if "internvl" in model_type:
            pixel_attrs = pixel_attrs.sum(dim=-3)
        
        elif "qwen" in model_type:
            pixel_attrs = pixel_attrs.sum(dim=-1)
        
        else:
            pixel_attrs = pixel_attrs.sum(dim=-3).sum(dim=1)
        

        return token_attrs, pixel_attrs
