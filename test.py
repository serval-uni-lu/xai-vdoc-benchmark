import torch
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import requests
from io import BytesIO

from qwen_vl_utils import process_vision_info

from functools import partial
from torch.nn import Dropout

from einops import rearrange

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward
from lxt.efficient import monkey_patch, monkey_patch_zennit

from zennit.composites import LayerMapComposite
import zennit.rules as z_rules

def load_qwen_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct"):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        #quantization_config=bnb_config,
        # low_cpu_mem_usage=True,
        dtype="auto",          # keep weights in fp16 after dequant chunks
        device_map="auto",                  # split across GPU/CPU automatically
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else None
        ),
    )
    return model, processor

def get_inputs(processor):
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" 
    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

    # Construct the multimodal conversation prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the main scene and the object being pointed at."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    return inputs, image

def get_target(inputs, model):
    inputs = inputs.to(model.device)
    model.eval()

    # 1) Decide a target sequence (reference or self-generated)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=32)
    # Use only the newly generated continuation as target
    target = gen[0, inputs['input_ids'].shape[1]:]          # tensor of target ids
    T = target.shape[0]
    return target, gen

def configure_lxt(model, use_zennit=False):

    attnLRP = {
        Qwen2_5_VLMLP: partial(patch_method, gated_mlp_forward),
        Qwen2RMSNorm: partial(patch_method, rms_norm_forward), 
        Dropout: partial(patch_method, dropout_forward),
        modeling_qwen2_5_vl: patch_attention,
    }

    monkey_patch(modeling_qwen2_5_vl, patch_map=attnLRP, verbose=True)

    zennit_comp = None

    if use_zennit:
        # Define rules for the Conv2d and Linear layers using 'zennit'
        conv_gamma = 100
        lin_gamma = 0.05
        # LayerMapComposite maps specific layer types to specific LRP rule implementations
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv3d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ])
        
        monkey_patch_zennit(verbose=True)

    # Set up the model for the explanation task
    # model.train()  # Switch to train mode to enable  gradient flow
    #model.gradient_checkpointing_enable()  # Optional: saves memory

    # Deactivate gradients on model parameters to save memory and ensure LRP rules apply
    for param in model.parameters():
        param.requires_grad = False

    if zennit_comp is not None:
        # Register the composite rules with the model
        zennit_comp.register(model)
    return zennit_comp



def get_relevance(model, inputs, zennit_comp=None):

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw

    # Text embeddings
    inputs_embeds = model.get_input_embeddings()(input_ids)
    # inputs_embeds.requires_grad_(True) # .to(model.device)

    # Vision embeddings
    # pixel_values.requires_grad_(True)
    image_embeds = model.get_image_features(pixel_values,
                                            image_grid_thw=image_grid_thw)
    
    image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    image_mask, _ = model.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    inputs_embeds = inputs_embeds.detach().requires_grad_(True)

    # inference and get the maximum logit at the last position (we can also explain other tokens)
    outputs = model(
                    #input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    #image_grid_thw=image_grid_thw,
                    #pixel_values=pixel_values,
                    #position_ids=position_ids,
                    use_cache=True
                    )
    
    output_logits = outputs["logits"]
    print(output_logits)
    max_logits, _ = torch.max(output_logits[0, -1, :], dim=-1)
    max_logits.backward()

    if zennit_comp is not None:
        # Remove the registered composite to prevent interference in future iterations
        zennit_comp.remove()

    relevance = (inputs_embeds.grad * inputs_embeds).float().sum(-1).detach().cpu()[0] # cast to float32 before summation for higher precision

    return relevance


def reshape_visual_relevance(model, processor, image_size):

    patch_size = processor.image_processor.patch_size
    spatial_merge_size = model.config.vision_config.spatial_merge_size

    height_new, width_new = smart_resize(
        width=image_size[0],
        height=image_size[1],
        factor=patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.size["shortest_edge"],
        max_pixels=processor.image_processor.size["longest_edge"],
    )

    n_patches_x = width_new // patch_size // spatial_merge_size
    n_patches_y = height_new // patch_size // spatial_merge_size
    return (n_patches_x, n_patches_y)


def visualize_image_relevance(image, img_relevance, figsize=(8,8), save_path=None):


    # Convert the image to an array
    img_array = np.array(image.convert("RGBA"))  # (height, width, channels)

    similarity_map_image = Image.fromarray((img_relevance.cpu().numpy() * 255).astype("uint8")).resize(
            image.size, Image.Resampling.BICUBIC
    )

    
    show_colorbar = False
    # Create the figure
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(img_array)
        im = ax.imshow(
            similarity_map_image,
            cmap=sns.color_palette("mako", as_cmap=True),
            alpha=0.5,
        )

        if show_colorbar:
            fig.colorbar(im)
        ax.set_axis_off()
        fig.tight_layout()

        # ---- SAVE FIGURE ----
        if save_path is None:
            save_path = "img/relevance_img_overlay.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def visualize_text_relevance(text, token_relevance, figsize=(8,8), save_path=None):

    # "Importance": abs by default (signed scores → magnitude)
    idx_max = int(torch.argmax(token_relevance))

    # Keep it readable: show top-k (optional)
    k = min(30, len(text))
    top_idx = torch.topk(token_relevance, k).indices.tolist()[::-1]  # highest → lowest
    top_text = [text[i] for i in top_idx]
    top_imp    = token_relevance[top_idx].numpy()

    # Colors: grey for all, one highlighted
    colors = ['lightgray'] * k
    colors[top_idx.index(idx_max)] = 'tab:red'  # color only the most important

    plt.figure(figsize=figsize)
    y = np.arange(k)
    plt.barh(y, top_imp, color=colors)
    plt.yticks(y, top_text)
    plt.gca().invert_yaxis()
    plt.xlabel('Token importance (|relevance|)')
    plt.title('Top-k token attributions (max highlighted)')
    plt.tight_layout()
    # ---- SAVE FIGURE ----
    if save_path is None:
        save_path = "img/relevance_prompt_overlay.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")



def main():

    model, processor = load_qwen_model(model_id="Qwen/Qwen2.5-VL-3B-Instruct")

    inputs, image = get_inputs(processor)

    # target, gen = get_target(inputs, model)

    zennit_comp = configure_lxt(model, use_zennit=False)

    relevance = get_relevance(model, inputs, zennit_comp=zennit_comp)
    
    # normalize relevance between [-1, 1] for plotting
    relevance_norm = relevance / relevance.abs().max()

    input_ids = inputs["input_ids"]
    img_mask = (input_ids == processor.image_token_id)
    token_mask = (input_ids != processor.image_token_id)
    img_mask = img_mask.cpu()
    token_mask = token_mask.cpu()

    idx = 0
    image_size = image.size
    n_patches = reshape_visual_relevance(model, processor, image_size)

    img_relevance = rearrange(
                relevance_norm[img_mask[idx]],  # (n_patches_x * n_patches_y, dim)
                "(h w) -> w h",
                w=n_patches[0],
                h=n_patches[1],
            )  # (n_patches_x, n_patches_y, dim)
    
    ids = input_ids[idx][token_mask[idx]]
    token_relevance = relevance_norm[token_mask[idx]]

    prompt_text = processor.batch_decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=True
    )

    visualize_image_relevance(image, img_relevance)
    visualize_text_relevance(prompt_text, token_relevance)



