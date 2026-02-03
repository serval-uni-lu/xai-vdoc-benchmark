from typing import Literal, Tuple, List

from PIL import Image
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
    
    def get_raw_attributions(self,
                             image,
                             question,
                             target_indices: List[int],
                             **kwargs,
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.wrapper.get_inputs(image, question)

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]  #.unsqueeze(0)

        text_embeds = self.wrapper.embed_text(input_ids)

        captum_forward = (text_embeds.requires_grad_(), pixel_values.requires_grad_())
        kwargs_dict = {k: v for k, v in inputs.items() if k not in ['pixel_values']}
        # captum_add_forward = (attention_mask, input_ids, return_probs, kwargs)
        captum_add_forward = (kwargs_dict)

        baselines = None
        use_baselines = False
        if self.xai_name in ["integrated"]:
            use_baselines = True
    
        if use_baselines:
            token_reference = TokenReferenceBase(reference_token_idx=self.wrapper.processor.tokenizer.pad_token_id)
            # generate reference for each sample
            reference_ids = token_reference.generate_reference(
                                    input_ids.shape[-1],
                                    device=self.device
                                    ).unsqueeze(0)
            reference_embeds = self.wrapper.embed_text(reference_ids)
            baselines = (reference_embeds, torch.zeros_like(pixel_values))
        
        pred_result = self.wrapper.predict(inputs=inputs,
                                           return_logits=False,
                                           **kwargs,
                                           )
        new_ids = pred_result["new_ids"]
        target_token = new_ids[target_indices].cpu().numpy().tolist()

        # Get attributions
        if use_baselines:
            token_attr, pixel_attr = self.explainer.attribute(inputs=captum_forward,
                                                            baselines=baselines,
                                                            target=target_token,
                                                            additional_forward_args=captum_add_forward,
                                                            n_steps=5,
                                                            internal_batch_size=1,
                                                            )
        else:
            token_attr, pixel_attr = self.explainer.attribute(inputs=captum_forward,
                                                target=target_token,
                                                additional_forward_args=captum_add_forward,
                                                )
        
        token_attr = token_attr.detach().cpu()
        pixel_attr = pixel_attr.detach().cpu()

        return token_attr, pixel_attr
