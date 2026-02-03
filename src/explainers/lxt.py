from typing import Dict, Any, List, Tuple

from PIL import Image
import torch
from zennit.composites import LayerMapComposite
import zennit.rules as z_rules
from lxt.efficient import monkey_patch, monkey_patch_zennit


from src.explainers import BaseExplainer
from src.models import BaseVLMWrapper
from src.models.factory import create_model_wrapper

class LXTExplainer(BaseExplainer):
    def __init__(self,
                model_wrapper: BaseVLMWrapper):
        super().__init__(model_wrapper)
        self.zennit_comp = None
        self.apply_monkey_patch(verbose=True,
                                use_zennit=False,
                                )
    
    def apply_monkey_patch(self,
                           verbose: bool = False,
                           use_zennit: bool = False,
                           ):
        
        module_root = self.wrapper.get_root_module()
        patch_dict = self.wrapper.get_patch_map()
        model_id = self.wrapper.model.config.name_or_path
        vlm_type = self.wrapper.model.config.model_type
        
        monkey_patch(module=module_root,
                     patch_map=patch_dict,
                     verbose=verbose)

        if use_zennit:
            monkey_patch_zennit(verbose=verbose)
        
        try:
            # 3. Load Model (It will now instantiate using PATCHED classes)
            # We use your factory from before
            print(f"[*] Loading patched model: {model_id}")
            wrapper = create_model_wrapper(model_id,
                                           vlm_type=vlm_type)
            
            self.wrapper = wrapper

        finally:
            print("[!] Exiting Patch Context.")
            print("[!] WARNING: Python classes are still modified in memory.")
            print("[!] It is highly recommended to restart the kernel/process before running a different XAI method.")
            # Unlike hooks, monkey patching classes is very hard to undo robustly 
            # without reloading the 'transformers' library entirely.
        
        # Configure LXT (Zennit etc.)
        # Assuming you have a helper for this
        zennit_comp = self.configure_lxt(wrapper.model,
                                        use_zennit=use_zennit,
                                        )
        self.zennit_comp = zennit_comp
        
    def configure_lxt(self,
                      model: torch.nn.Module,
                      use_zennit=False):

        zennit_comp = None

        if use_zennit:
            # Define rules for the Conv2d and Linear layers using 'zennit'
            conv_gamma = 100
            lin_gamma = 0.05
            # LayerMapComposite maps specific layer types to specific LRP rule implementations
            zennit_comp = LayerMapComposite([
                (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
                (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
            ])
            
            # monkey_patch_zennit(verbose=True)

        # Set up the model for the explanation task
        model.train()  # Switch to train mode to enable  gradient flow
        model.gradient_checkpointing_enable()  # Optional: saves memory

        # Deactivate gradients on model parameters to save memory and ensure LRP rules apply
        for param in model.parameters():
            param.requires_grad = False

        if zennit_comp is not None:
            # Register the composite rules with the model
            zennit_comp.register(model)
        return zennit_comp

    def get_raw_attributions(self,
                             image,
                             question: str,
                             target_indices: List[int],
                             **kwargs
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        inputs = self.wrapper.get_inputs(image, question)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        kwargs_dict = {k: v for k, v in inputs.items() if k not in ['pixel_values']}

        text_embeds = self.wrapper.embed_text(input_ids)
        text_embeds.requires_grad_(True)
        pixel_values.requires_grad_(True)

        log_logits = self.wrapper(text_embeds, pixel_values,
                                  return_probs=True,
                                  **kwargs_dict,
                                  )
        max_logits, _ = torch.max(log_logits, dim=-1)
        max_logits.backward()

        if self.zennit_comp is not None:
            # Remove the registered composite to prevent interference in future iterations
            self.zennit_comp.remove()

        # if full_relevance:
        relevance_img = (pixel_values.grad * pixel_values).float().detach().cpu()
        #relevance_img_norm = relevance_img / relevance_img.abs().max()

        relevance_text = (text_embeds.grad * text_embeds).float().sum(-1).detach().cpu()

        return relevance_text, relevance_img
    
