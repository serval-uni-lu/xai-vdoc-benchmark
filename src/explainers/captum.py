from typing import Literal, Tuple, List

import numpy as np
import cv2
from PIL import Image
import torch
from captum.attr import (
    IntegratedGradients, InputXGradient, Saliency,
    TokenReferenceBase,
    visualization
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
    
    def get_raw_attributions(self,
                             image,
                             question,
                             target_indices: List[int],
                             **kwargs,
                            ) -> Tuple[torch.Tensor]:
        inputs = self.wrapper.get_inputs(image, question)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        pixel_values = inputs.pixel_values
        return_probs = True

        text_embeds = self.wrapper.embed_text(input_ids)

        captum_forward = (text_embeds.requires_grad_(), pixel_values.requires_grad_())
        captum_add_forward = (attention_mask, input_ids, return_probs, kwargs)

        baselines = None
        use_baselines = False
        if self.xai_name in ["integrated"]:
            use_baselines = True
    
        if use_baselines:
            token_reference = TokenReferenceBase(reference_token_idx=self.wrapper.processor.tokenizer.pad_token_id)
            # generate reference for each sample
            reference_ids = token_reference.generate_reference(
                                    input_ids.shape[-1],
                                    device=self.device).unsqueeze(0)
            reference_embeds = self.wrapper.embed_text(reference_ids)
            baselines = (reference_embeds, pixel_values * 0.0)
        
        pred_result = self.wrapper.predict(input_ids=input_ids,
                                           pixel_values=pixel_values,
                                           attention_mask=attention_mask,
                                           return_logits=False,
                                           **kwargs,
                                           )
        new_ids = pred_result["new_ids"]
        target_token = new_ids[target_indices].cpu().numpy().tolist()

        # Get attributions
        if use_baselines:
            attributions = self.explainer.attribute(inputs=captum_forward,
                                                baselines=baselines,
                                                target=target_token,
                                                additional_forward_args=captum_add_forward,
                                                n_steps=10)
        else:
            attributions = self.explainer.attribute(inputs=captum_forward,
                                                target=target_token,
                                                additional_forward_args=captum_add_forward,
                                                )

        # if isinstance(target_tokens, list):
        #     # Average over all attributions

        #     for idx, target_token in enumerate(target_tokens):     
        #         # Get attributions
        #         if use_baselines:
        #             attr_token_i = self.explainer.attribute(inputs=captum_forward,
        #                                                 baselines=baselines,
        #                                                 target=target_token,
        #                                                 additional_forward_args=captum_add_forward,
        #                                                 n_steps=10)
        #         else:
        #             attr_token_i = self.explainer.attribute(inputs=captum_forward,
        #                                                 target=target_token,
        #                                                 additional_forward_args=captum_add_forward,
        #                                                 )
        return attributions
