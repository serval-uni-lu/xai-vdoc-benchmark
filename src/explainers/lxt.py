from typing import Dict, Any, List, Tuple, Optional

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
                model_wrapper: BaseVLMWrapper,
                token_wise = True,
                ):
        # super().__init__(model_wrapper)
        self.wrapper = model_wrapper
        self.zennit_comp = None
        self.token_wise = token_wise

        # Check & Apply Patches (Idempotent)
        # This will update self.wrapper if a reload occurred
        self._ensure_patched(verbose=True,
                            use_zennit=False,
                            )

        # Update parent with the potentially new wrapper
        super().__init__(self.wrapper)


    def _ensure_patched(self,
                           verbose: bool = False,
                           use_zennit: bool = False,
                           ):
        
        # Get Patch Info from the wrapper (Delegated to wrapper logic)
        if not hasattr(self.wrapper, "get_patch_map"):
            print("[!] Wrapper does not support LXT patching auto-detection.")
            return
        
        
        module_root = self.wrapper.get_root_module()
        patch_dict = self.wrapper.get_patch_map()
        model_id = self.wrapper.model.config.name_or_path
        vlm_type = self.wrapper.model.config.model_type

        # --- IDEMPOTENCY CHECK ---
        # We check a flag on the module itself to see if we already touched it
        if getattr(module_root, "__lxt_is_patched__", False):
            # Already patched! Do not reload model.
            return 
        
        print(f"[*] Applying LXT Monkey Patches to {module_root}...")
        monkey_patch(module=module_root,
                     patch_map=patch_dict,
                     verbose=verbose)

        if use_zennit:
            monkey_patch_zennit(verbose=verbose)
        
        # Mark as patched
        setattr(module_root, "__lxt_is_patched__", True)
        
        try:
            # 3. Load Model (It will now instantiate using PATCHED classes)
            # We use your factory from before
            print(f"[*] Reloading patched model: {model_id}")
            device = self.wrapper.device
            del self.wrapper
            
            self.wrapper = create_model_wrapper(model_id,
                                           vlm_type=vlm_type)
            self.wrapper.to_device(device)
            torch.cuda.empty_cache()
            
        finally:
            print("[!] Exiting Patch Context.")
            print("[!] WARNING: Python classes are still modified in memory.")
            print("[!] It is highly recommended to restart the kernel/process before running a different XAI method.")
            # Unlike hooks, monkey patching classes is very hard to undo robustly 
            # without reloading the 'transformers' library entirely.
        
        # Configure LXT (Zennit etc.)
        # Assuming you have a helper for this
        zennit_comp = self._configure_lxt(self.wrapper.model,
                                        use_zennit=use_zennit,
                                        )
        self.zennit_comp = zennit_comp
        
    def _configure_lxt(self,
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
                             text: str,
                             target_indices: Optional[int | List[int]] = None,
                             # token_wise: bool = True,
                             **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. SETUP PRISTINE INPUTS
        inputs = self.wrapper.get_inputs(image, text)

        pred_results = kwargs.get("pred_results", None)
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs=inputs,
                                            return_logits=False,
                                            **kwargs,
                                            )
        full_ids = pred_results["full_ids"]

        t_start = inputs["input_ids"].shape[1]
        t_end = full_ids.shape[-1]
        gen_len = t_end - t_start

        # Default to explaining all generated tokens if none specified
        if target_indices is None:
            target_indices = list(range(gen_len))

        # Build full sequence inputs
        inputs["input_ids"] = full_ids.clone().unsqueeze(0)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])


        pixel_values = inputs["pixel_values"].clone().detach()
        pixel_values.requires_grad_(True)
        
        kwargs_dict = {k: v for k, v in inputs.items() if k not in ['pixel_values']}

        # We must get the embeddings for the FULL sequence to pass to Captum/LRP
        text_embeds = self.wrapper.embed_text(inputs["input_ids"]).clone().detach()
        text_embeds.requires_grad_(True)

        # 2. RUN FORWARD PASS
        # (Zennit composite should already be registered on the model from _ensure_patched)
        log_logits = self.wrapper.get_captum_forward(
            text_embeds=text_embeds,
            pixel_values=pixel_values,
            kwargs_dict=kwargs_dict,
            return_full_sequence=True
        )

        # 3. CAUSAL GATHER
        # log_logits shape: [1, seq_len, vocab_size]
        gen_logits = log_logits[:, t_start-1 : t_end-1, :]
        gen_ids_1d = full_ids.squeeze()[t_start:t_end].to(dtype=torch.long, device=gen_logits.device)
        generated_token_ids = gen_ids_1d.unsqueeze(0).unsqueeze(-1)
        
        # Gather the specific logits of the generated words!
        # target_logits shape: [1, gen_len]
        target_logits = gen_logits.gather(dim=-1, index=generated_token_ids).squeeze(-1)

        token_attributions = []
        pixel_attributions = []

        # 4. BACKWARD PASS ENGINE (Gradient * Activation)
        if not self.token_wise:
            # --- SENTENCE-LEVEL ---
            self.wrapper.model.zero_grad()
            
            # Sum the target logits for the requested indices
            sentence_score = target_logits[0, target_indices].sum()
            sentence_score.backward(retain_graph=True)
            
            # LRP Rule: Relevance = Gradient * Activation
            rel_img = (pixel_values.grad * pixel_values).float().sum(-1).detach().cpu()
            rel_text = (text_embeds.grad * text_embeds).float().sum(-1).detach().cpu()
            
            # Store
            token_attributions.append(rel_text)
            pixel_attributions.append(rel_img)
            
        else:
            # --- TOKEN-WISE ---
            for idx in target_indices:
                # Clear gradients from the previous token!
                self.wrapper.model.zero_grad()
                if pixel_values.grad is not None: pixel_values.grad.zero_()
                if text_embeds.grad is not None: text_embeds.grad.zero_()
                
                # Backpropagate just this specific token
                target_logits[0, idx].backward(retain_graph=True)
                
                # LRP Rule: Relevance = Gradient * Activation
                rel_img = (pixel_values.grad * pixel_values).float().sum(-1).detach().cpu()
                rel_text = (text_embeds.grad * text_embeds).float().sum(-1).detach().cpu()
                
                token_attributions.append(rel_text)
                pixel_attributions.append(rel_img)

        # 5. CLEANUP & FORMATTING
        if self.zennit_comp is not None:
            self.zennit_comp.remove()
            self.zennit_comp = None # Ensure it doesn't get removed twice
            
        torch.cuda.empty_cache()

        # Stack into [len(target_indices), num_pixels] tensors
        relevance_img = torch.stack(pixel_attributions, dim=0)
        relevance_text = torch.cat(token_attributions, dim=0)

        # Slice the text relevance to only include the prompt if desired, 
        # or leave as full sequence length.
        relevance_text = relevance_text[:, :t_start]
        
        return relevance_text, relevance_img

