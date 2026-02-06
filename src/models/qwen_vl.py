import importlib
from types import ModuleType
from typing import Any, Dict
from functools import partial

import torch
import torch.nn as nn
from torch.nn import Dropout
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from lxt.efficient.patches import patch_method, patch_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, dropout_forward

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

        if pixel_values.dim() == 3:
            batch_size = pixel_values.shape[0]
            
            # 1. Flatten pixel_values back to 2D for Qwen
            # (Batch, Patches, Dim) -> (Batch*Patches, Dim)
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
            
            # 2. Expand image_grid_thw to match the batch size
            # Current grid shape: (1, 3) -> We need (Batch_Size, 3)
            if image_grid_thw.shape[0] == 1 and batch_size > 1:
                image_grid_thw = image_grid_thw.repeat(batch_size, 1)

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
    
    def get_root_module(self) -> ModuleType:
        return importlib.import_module("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl")

    def get_patch_map(self) -> Dict[str, Any]:
        attnLRP = {
                Qwen2_5_VLMLP: partial(patch_method, gated_mlp_forward),
                Qwen2RMSNorm: partial(patch_method, rms_norm_forward),
                Dropout: partial(patch_method, dropout_forward),
                modeling_qwen2_5_vl: patch_attention,
            }
        return attnLRP

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

    def get_tam_config(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        image_grid_thw = inputs.get('image_grid_thw')
        if image_grid_thw is None:
             # Fallback if batching stripped it, or default size
             raise ValueError("TAM requires image_grid_thw for Qwen")
        
        vision_shape = (image_grid_thw[0, 1] // 2, image_grid_thw[0, 2] // 2)
        
        special_ids = {'img_id': [151652, 151653],
                        'prompt_id': [151653, [151645, 198, 151644, 77091]], 
                        'answer_id': [[198, 151644, 77091, 198], -1]}
        
        return {
            "vision_shape": vision_shape,
            "special_ids": special_ids,
            #"vis_inputs": vis_inputs
        }