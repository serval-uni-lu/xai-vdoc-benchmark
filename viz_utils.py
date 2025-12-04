import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from captum.attr import visualization

def normalize_similarity_map(similarity_map: torch.Tensor,
                             value_range=None,
                             eps: float = 1e-8) -> torch.Tensor:
    """
    Min-max normalize a similarity / attribution map to [0, 1].

    similarity_map: torch.Tensor of arbitrary shape (e.g. [H, W])
    value_range: (min, max) or None. If None, use min/max from the tensor.
    """
    if value_range is None:
        vmin = similarity_map.min()
        vmax = similarity_map.max()
    else:
        vmin, vmax = value_range

    denom = (vmax - vmin).clamp(min=eps)
    return (similarity_map - vmin) / denom

def prepare_for_visualization(original_image, pixel_attr, image_grid_thw,
                              patch_size=14, figsize=(12, 4)):
        grid_t, grid_h, grid_w = image_grid_thw.tolist()  # e.g. [1, H_patches, W_patches]

        pixel_attr = pixel_attr.view(grid_t, grid_h, grid_w, -1)[0]
        pixel_attr = pixel_attr.mean(axis=-1)
        # pixel_attr = (pixel_attr - pixel_attr.min()) / (pixel_attr.max() - pixel_attr.min() + 1e-8)

        resized_height = grid_h * patch_size
        resized_width  = grid_w * patch_size

         # resize original image to match the grid resolution
        img_resized = original_image.resize((resized_width, resized_height), Image.BILINEAR) # type: ignore
        img_resized_np = np.asarray(img_resized).astype(np.float32) / 255.0  # [H, W, 3]

        # # upsample heatmap from [grid_h, grid_w] → [resized_H, resized_W]
        # heat = pixel_attr.unsqueeze(0).unsqueeze(0)  # [1, 1, H_p, W_p]
        # hidden_len = heat.shape[-1]
        # heat_up = F.interpolate(
        #     heat,
        #     size=(resized_height*resized_width, hidden_len),
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze().cpu().numpy()

         # --- 2. normalize to [0, 1] ---
        heat_2d = normalize_similarity_map(pixel_attr, value_range=None)
        heat_2d_np = heat_2d.to(torch.float32).cpu().numpy()       # [H_p, W_p]

        # --- 3. to uint8 and PIL image (grayscale) ---
        heat_2d_uint8 = (heat_2d_np * 255).astype("uint8")
        heat_pil = Image.fromarray(heat_2d_uint8, mode="L")

        # --- 4. resize heatmap to original image size using PIL (your approach) ---
        heat_pil_resized = heat_pil.resize(original_image.size, Image.Resampling.BICUBIC)

        heat_up = np.asarray(heat_pil_resized).astype(np.float32) / 255.0  # [H, W]

        # --- 5. Use Captum's visualize_image_attr_multiple ---
        visualization.visualize_image_attr_multiple(
            np.expand_dims(heat_up, 2),
            img_resized_np,
            methods=["original_image", "heat_map", "blended_heat_map"],
            signs=["all", "absolute_value", "absolute_value"],
            show_colorbar=True,
            titles=["Original", "Attribution", "Overlay"],
            fig_size=figsize
        )



def show_side_by_side(original_image, pixel_values, pixel_attr, image_grid_thw,
                      patch_size=14, cmap="gray",figsize=(12, 4)):
    grid_t, grid_h, grid_w = image_grid_thw

    # build coarse map from pixel_values (same as gray version)
    pv_attr = pixel_attr.reshape(grid_t, grid_h, grid_w, -1)[0]
    heat_2d = pv_attr.mean(axis=-1)
    heat_2d = (heat_2d - heat_2d.min()) / (heat_2d.max() - heat_2d.min() + 1e-8)

    # build coarse map from pixel_values (same as gray version)
    pv = pixel_values.reshape(grid_t, grid_h, grid_w, -1)[0]
    pv = pv.mean(axis=-1)
    pv = (pv - pv.min()) / (pv.max() - pv.min() + 1e-8)

    # size of the resized image inside the processor
    resized_H = grid_h * patch_size
    resized_W = grid_w * patch_size

    # resize original image to match
    img_resized = original_image.resize((resized_W, resized_H))
    

    _, axs = plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(img_resized)
    axs[0].set_title("Original (resized)")
    axs[0].axis("off")

    axs[1].imshow(pv, cmap=cmap)
    axs[1].set_title("pixel_values (mean per patch)")
    axs[1].axis("off")

    axs[2].imshow(heat_2d, cmap=cmap)
    axs[2].set_title("pixel_values attributions (mean per patch)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def show_pixel_attribution_captum(pixel_values, pixel_attr,
                                  image_grid_thw, cmap="gray",
                                  figsize=(12, 4)):
    grid_t, grid_h, grid_w = image_grid_thw

    # build coarse map from pixel_values (same as gray version)
    pv_attr = pixel_attr.reshape(grid_t, grid_h, grid_w, -1)[0]
    # heat_2d = pv_attr.mean(axis=-1)
    # heat_2d = (heat_2d - heat_2d.min()) / (heat_2d.max() - heat_2d.min() + 1e-8)

    # build coarse map from pixel_values (same as gray version)
    pv = pixel_values.reshape(grid_t, grid_h, grid_w, -1)[0]
    pv = pv.mean(axis=-1).unsqueeze(2)
    # pv = (pv - pv.min()) / (pv.max() - pv.min() + 1e-8)

    # size of the resized image inside the processor
    # resized_H = grid_h * patch_size
    # resized_W = grid_w * patch_size
    print(pv.shape, pv_attr.shape)


# --- 5. Use Captum's visualize_image_attr_multiple ---
    visualization.visualize_image_attr_multiple(
        pv_attr.cpu().detach().numpy(),
        pv.cpu().detach().numpy(),
        methods=["original_image", "heat_map", "blended_heat_map"],
        signs=["all", "all", "all"],
        show_colorbar=True,
        cmap=cmap,
        titles=["Original", "Attribution", "Overlay"],
        fig_size=figsize,
    )
