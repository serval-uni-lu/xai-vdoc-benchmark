from typing import Dict, Any, List, Tuple, Optional

from PIL import Image
import numpy as np
import torch

from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from src.explainers.tam_utils.tam import get_attributions, TAM
from src.explainers.utils import align_llm_visuals_to_pixels

class TAMExplainer(BaseExplainer):
    def __init__(self,
                 model_wrapper: BaseVLMWrapper):
        super().__init__(model_wrapper)

    def get_raw_attributions(self,
                            image,
                            text: str,
                            target_indices: Optional[int | List[int]],
                            **kwargs
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        

        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results", None)
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs,
                                                return_logits=True,
                                                )
        
        logits = pred_results["logits"]
        generated_ids = pred_results["full_ids"].cpu()
        # new_ids = pred_results["new_ids"]
        # pred_text = pred_results["text"]

        tam_config = self.wrapper.get_tam_config(inputs)
        
        vis_inputs = np.array(image.convert("RGB"))

        #target_token = new_ids[target_indices].cpu().numpy().tolist()        

        # Get attributions
        img_scores_list = []
        text_attributions = []
        img_attributions = []
        for i in range(len(logits)):
            # result = get_attributions(
            #             generated_ids.cpu().tolist(),        
            #             tam_config["vision_shape"],
            #             logits,
            #             tam_config["special_ids"],
            #             vis_inputs,
            #             self.wrapper.processor,
            #             target_token_idx=i,
            #             img_scores_list=img_scores_list,
            #             eval_only=True)

            result = TAM(
                        tokens=generated_ids.cpu().tolist(),        
                        vision_shape=tam_config["vision_shape"],
                        logit_list=logits,
                        special_ids=tam_config["special_ids"],
                        vision_input=vis_inputs,
                        processor=self.wrapper.processor,
                        save_fn="",
                        target_token=i,
                        img_scores_list=img_scores_list,
                        eval_only=True,
                        return_components=True,
                        )
        
            img_attribution = result["img_map_norm"]
            text_attribution = result["txt_scores_raw"]
            #print(text_attribution.shape)
        
            text_attributions.append(text_attribution)
            img_attributions.append(img_attribution)
        #img_attributions = img_attributions[-1]
        
        img_attributions = [torch.from_numpy(_).float() for _ in img_attributions]
        text_attributions = [torch.from_numpy(_).float() for _ in text_attributions]
        img_attributions = torch.stack(img_attributions)
        img_attributions = img_attributions.reshape(img_attributions.shape[0], -1)
        text_attributions = torch.stack(text_attributions)

        img_attributions = align_llm_visuals_to_pixels(img_attributions, inputs)

        return text_attributions, img_attributions
