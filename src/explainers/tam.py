from typing import Dict, Any, List, Tuple, Optional

from PIL import Image
import numpy as np
import torch

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from .tam_utils.tam import get_attributions

class TAMExplainer(BaseExplainer):
    def __init__(self,
                 model_wrapper: BaseVLMWrapper):
        super().__init__(model_wrapper)

    def get_raw_attributions(self,
                            image,
                            question: str,
                            target_indices: Optional[int | List[int]],
                            **kwargs
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        

        inputs = self.wrapper.get_inputs(image, question)

        pred_results = self.wrapper.predict(inputs,
                                            return_logits=True,
                                            )
        
        logits = pred_results["logits"]
        generated_ids = pred_results["full_ids"].cpu()
        new_ids = pred_results["new_ids"]
        pred_text = pred_results["decoded_text"]

        tam_config = self.wrapper.get_tam_config(inputs)
        
        vis_inputs = np.array(image.convert("RGB"))

        target_token = new_ids[target_indices].cpu().numpy().tolist()

        # Get attributions
        result = get_attributions(
                    generated_ids[0].cpu().tolist(),        
                    tam_config["vision_shape"],
                    logits,
                    tam_config["special_ids"],
                    vis_inputs,
                    self.wrapper.processor,
                    target_token_idx=target_token,
                    img_scores_list=[],
                    eval_only=True)
        
        img_attribution = result["img_map_norm"]
        text_attribution = result["prompt_scores_raw"]


        return text_attribution, img_attribution
