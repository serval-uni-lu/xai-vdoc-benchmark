from typing import Any, Dict
import torch
import torch.nn as nn

from src.models.base import BaseVLMWrapper

class QwenVL_Wrapper(BaseVLMWrapper):
    def __init__(self, model: nn.Module, processor: Any):
        super().__init__(model, processor)

    def embed_text(self, input_ids: torch.Tensor
                   ) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)
        
    
    def embed_images(self, pixel_values: torch.Tensor,
                    **kwargs,
                    ) -> torch.Tensor:
        image_grid_thw = kwargs.get("image_grid_thw", None)
        if image_grid_thw is None:
            raise ValueError("QwenVl needs image_grid_thw tensor")
        image_embeds = self.model.get_image_features(pixel_values,
                                                image_grid_thw=image_grid_thw)
        return image_embeds

    def merge_embeddings(self,
                        text_embeds: torch.Tensor,
                        image_embeds: torch.Tensor,
                        input_ids: torch.Tensor,
                        ) -> torch.Tensor:
        
        image_embeds = torch.cat(image_embeds,
                                dim=0
                                ).to(text_embeds.device, text_embeds.dtype)
        image_mask, _ = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds=text_embeds, image_features=image_embeds
        )
        inputs_embeds = text_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds
    
    def get_inputs(self, image, question) -> Dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        return inputs
