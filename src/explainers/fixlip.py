import numpy as np
import shapiq
import torch
import torch.nn.functional as F

from src.explainers import BaseExplainer
from src.explainers.fixlip_utils.fixlip import FIxLIP
from src.models import BaseVLMWrapper

# Assuming pred_probs is available in your repository
from src.utils.faithfulness_utils import pred_probs


class GenerativeVLMGame(shapiq.Game):
    """
    A shapiq.Game implementation for modern Generative VLMs (LLaVA, Qwen-VL, InternVL).
    Uses Teacher Forcing to evaluate multi-token generative sequences highly efficiently.
    Includes Patch Grouping (Super-Patches) to reduce player count for stable Shapley math.
    """

    def __init__(
        self,
        model_wrapper,
        image,
        text: str,
        target_ids: torch.Tensor,
        target_seq_index: int,
        batch_size: int = 8,
        mask_value: float = 0.0,
        patch_group_size: int = 1,  # NEW: 1 = no grouping. 2 = 2x2 blocks, 4 = 4x4 blocks.
    ):
        self.wrapper = model_wrapper
        self.device = model_wrapper.device
        self.batch_size = batch_size
        self.mask_value = mask_value
        self.patch_group_size = patch_group_size

        # 1. Base Inputs
        self.base_inputs = self.wrapper.get_inputs(image, text)
        self.input_ids = self.base_inputs["input_ids"]
        self.target_ids_expanded = target_ids.unsqueeze(0)

        self.target_seq_index = target_seq_index
        self.target_token_id = target_ids[target_seq_index].item()

        # 2. Architecture Setup
        self.model_type = getattr(self.wrapper.model.config, "model_type", "").lower()

        base_pixels = self.base_inputs["pixel_values"]
        if (
            "internvl" in self.model_type
            and base_pixels.ndim == 4
            or "qwen" in self.model_type
            and base_pixels.ndim == 2
            or "llava" in self.model_type
            and base_pixels.ndim == 3
        ):
            base_pixels = base_pixels.unsqueeze(0)
        self.base_inputs["pixel_values"] = base_pixels

        # --- NEW: COUNT IMAGE PLAYERS WITH GROUPING ---
        if "internvl" in self.model_type:
            # InternVL groups whole tiles. We ignore patch_group_size here.
            self.n_players_image = base_pixels.shape[1]

        elif "qwen" in self.model_type:
            # Qwen passes a flat list of patches. We need the 2D grid shape.
            if "image_grid_thw" in self.base_inputs:
                grid_thw = self.base_inputs["image_grid_thw"][0].cpu().numpy().tolist()
                self.grid_h, self.grid_w = grid_thw[1], grid_thw[2]
            else:
                # Fallback estimate if grid isn't available
                total_patches = base_pixels.shape[1]
                self.grid_h = self.grid_w = int(np.sqrt(total_patches))

            self.grouped_h = max(1, self.grid_h // self.patch_group_size)
            self.grouped_w = max(1, self.grid_w // self.patch_group_size)
            self.n_players_image = self.grouped_h * self.grouped_w

        else:  # LLaVA
            _, _, H, W = base_pixels.shape
            self.grid_h, self.grid_w = H, W
            self.grouped_h = max(1, H // self.patch_group_size)
            self.grouped_w = max(1, W // self.patch_group_size)
            self.n_players_image = self.grouped_h * self.grouped_w

        # Count Text Players
        valid_mask = torch.ones_like(self.input_ids[0], dtype=torch.bool)
        if self.wrapper.special_token_ids is not None:
            for skip_id in self.wrapper.special_token_ids:
                valid_mask &= self.input_ids[0] != skip_id

        self.valid_token_indices = torch.where(valid_mask)[0]
        self.n_players_text = len(self.valid_token_indices)

        # 3. Compute Normalization Value
        empty_mask_img = np.zeros((1, self.n_players_image), dtype=np.int32)
        empty_mask_txt = np.zeros((1, self.n_players_text), dtype=np.int32)
        empty_value = self._evaluate_masks(empty_mask_img, empty_mask_txt)[0]

        full_mask_img = np.ones((1, self.n_players_image), dtype=np.int32)
        full_mask_txt = np.ones((1, self.n_players_text), dtype=np.int32)
        full_value = self._evaluate_masks(full_mask_img, full_mask_txt)[0]

        super().__init__(
            n_players=self.n_players_image + self.n_players_text, normalize=True, normalization_value=empty_value
        )
        self.empty_value = empty_value
        self.full_value = full_value

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        p_mask = coalitions[:, : self.n_players_image]
        t_mask = coalitions[:, self.n_players_image :]
        return self._evaluate_masks(p_mask, t_mask)

    def value_function_crossmodal(self, coalitions_image: np.ndarray, coalitions_text: np.ndarray) -> np.ndarray:
        n_img_masks = coalitions_image.shape[0]
        n_txt_masks = coalitions_text.shape[0]
        combined_p_mask = np.repeat(coalitions_image, n_txt_masks, axis=0)
        combined_t_mask = np.tile(coalitions_text, (n_img_masks, 1))
        flat_values = self._evaluate_masks(combined_p_mask, combined_t_mask)
        return flat_values.reshape(n_img_masks, n_txt_masks)

    def _evaluate_masks(self, p_mask_np: np.ndarray, t_mask_np: np.ndarray) -> np.ndarray:
        budget = p_mask_np.shape[0]
        game_values = np.zeros(budget, dtype=np.float32)

        p_mask_all = torch.tensor(p_mask_np, device=self.device, dtype=torch.int32)
        t_mask_all = torch.tensor(t_mask_np, device=self.device, dtype=torch.int32)

        for batch_start in range(0, budget, self.batch_size):
            current_bs = min(self.batch_size, budget - batch_start)
            batch_end = batch_start + current_bs

            p_mask = p_mask_all[batch_start:batch_end]
            t_mask = t_mask_all[batch_start:batch_end]

            masked_kwargs, new_input_ids, masked_pixels = self._apply_batched_masks(p_mask, t_mask, current_bs)

            with torch.no_grad():
                log_probs = pred_probs(
                    model=self.wrapper,
                    inputs=masked_kwargs,
                    new_input_ids=new_input_ids,
                    pixel_values=masked_pixels,
                    output_ids=self.target_ids_expanded.expand(current_bs, -1),
                )
                probs = torch.exp(log_probs)

            game_values[batch_start:batch_end] = probs[:, self.target_seq_index].detach().float().cpu().numpy()

        return game_values

    def _apply_batched_masks(self, p_mask, t_mask, bs):
        # 1. Text Masking
        # FIX: Ensure we use self.input_ids, not self.base_inputs
        input_ids = self.input_ids.expand(bs, -1).clone()
        pad_id = self.wrapper.processor.tokenizer.pad_token_id or 0

        idx_expanded = self.valid_token_indices.unsqueeze(0).expand(bs, -1)
        replace_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        replace_mask.scatter_(1, idx_expanded, t_mask == 0)
        input_ids[replace_mask] = pad_id

        target_ids_bs = self.target_ids_expanded.expand(bs, -1)
        new_input_ids = torch.cat([input_ids, target_ids_bs], dim=1)

        # 2. Image Masking
        base_pixels = self.base_inputs["pixel_values"]
        pixel_values = base_pixels.expand(bs, *base_pixels.shape[1:]).clone()

        if "internvl" in self.model_type:
            # InternVL uses tiles, grouping is ignored here
            expanded_mask = p_mask.view(bs, -1, 1, 1, 1)

        else:
            # --- NEW: UPSAMPLE THE SMALL MASK TO FULL RESOLUTION ---
            if self.patch_group_size > 1:
                # 1. Reshape the small super-patch mask into a 2D grid
                small_mask_grid = p_mask.view(bs, 1, self.grouped_h, self.grouped_w).float()

                # 2. Stretch it out to match the original grid size using Nearest Neighbor
                upsampled_mask = F.interpolate(small_mask_grid, size=(self.grid_h, self.grid_w), mode="nearest").int()
            else:
                upsampled_mask = p_mask.view(bs, 1, self.grid_h, self.grid_w)

            # --- Apply the full resolution mask to the tensors ---
            if "qwen" in self.model_type:
                # Qwen expects flat (bs, num_patches, patch_dim)
                expanded_mask = upsampled_mask.view(bs, -1, 1)
            else:  # LLaVA
                # LLaVA expects spatial (bs, C, H, W)
                _, C, _, _ = pixel_values.shape
                expanded_mask = upsampled_mask.expand(-1, C, -1, -1)

        pixel_values = torch.where(
            expanded_mask == 1,
            pixel_values,
            torch.tensor(self.mask_value, device=self.device, dtype=pixel_values.dtype),
        )

        # 3. Additional Kwargs
        masked_kwargs = {
            k: v.repeat(bs, 1) if k == "image_grid_thw" else v
            for k, v in self.base_inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }

        return masked_kwargs, new_input_ids, pixel_values


class FIxLIPExplainer(BaseExplainer):
    """
    Computes Shapley Interaction values using the official FIxLIP repository.
    Leverages GenerativeVLMGame for high-speed batched evaluation.
    """

    def __init__(
        self,
        model_wrapper: BaseVLMWrapper,
        budget: int = 2048,  # Higher budget recommended for Cross-Modal XGBoost
        batch_size: int = 8,
        mask_value: float = 0.0,
        approximation_type: str = "proxyshap",
        seed: int = 42,
        patch_group_size: int = 1,  # Set to 2 or 4 to group patches and speed up math!
    ):
        super().__init__(model_wrapper)
        self.budget = budget
        self.batch_size = batch_size
        self.mask_value = mask_value
        self.seed = seed
        self.patch_group_size = patch_group_size
        self.approximation_type = approximation_type

    def _attribute(
        self,
        image,
        text: str,
        target_indices: int | list[int] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the SHAP explanation loop.
        Returns:
            1. token_shap_tensor: (num_targets, seq_len)
            2. pixel_attributions: (num_targets, H, W)
            3. synergy_attributions: (num_targets, H, W, seq_len)
        """
        # 1. Base Forward Pass to get target IDs & sequence length
        inputs = self.wrapper.get_inputs(image, text)
        seq_len = inputs["input_ids"].shape[1]

        pred_results = kwargs.get("pred_results")
        if pred_results is None:
            pred_results = self.wrapper.predict(inputs, return_logits=True, **kwargs)

        new_ids = pred_results["new_ids"]
        if new_ids.dim() > 1:
            new_ids = new_ids[0]

        indices_to_compute = (
            target_indices
            if isinstance(target_indices, list)
            else [target_indices]
            if target_indices is not None
            else list(range(len(new_ids)))
        )
        model_type = getattr(self.wrapper.model.config, "model_type", "").lower()

        # 2. Setup SHAP Accumulators
        pixel_shap = []
        token_shap = []
        cross_modal_synergies = []

        # 3. Execution Loop (One Game per Target Token)
        for _, real_idx in enumerate(indices_to_compute):
            # --- Initialize our Custom Batched Game ---
            game = GenerativeVLMGame(
                model_wrapper=self.wrapper,
                image=image,
                text=text,
                target_ids=new_ids,
                target_seq_index=real_idx,
                batch_size=self.batch_size,
                mask_value=self.mask_value,
                patch_group_size=self.patch_group_size,
            )

            # --- Initialize Official FIxLIP ---
            fixlip_approx = FIxLIP(
                n_players_image=game.n_players_image,
                n_players_text=game.n_players_text,
                mode="banzhaf",
                max_order=2,
                random_state=self.seed,
            )

            # --- Run Cross-Modal XGBoost Approximation ---
            ivs = fixlip_approx.approximate_crossmodal(game=game, budget=self.budget, approximation_type="proxyshap")

            # ==========================================
            # STEP A: EXTRACT RAW DATA
            # ==========================================
            # 1. Raw Order 1 Marginals
            raw_p_s = np.array([ivs[(i,)] for i in range(game.n_players_image)])
            raw_t_s = np.array([ivs[(game.n_players_image + j,)] for j in range(game.n_players_text)])

            # 2. Raw Order 2 Cross-Modal Synergy Matrix: (n_images, n_texts)
            interaction_matrix = np.zeros((game.n_players_image, game.n_players_text))
            for img_idx in range(game.n_players_image):
                for txt_idx in range(game.n_players_text):
                    global_txt_idx = game.n_players_image + txt_idx
                    interaction_matrix[img_idx, txt_idx] = ivs[(img_idx, global_txt_idx)]

            # ==========================================
            # STEP B: CALCULATE TOTAL 1D IMPORTANCE
            # ==========================================
            # Add the absolute Marginal to the absolute sum of all Synergies
            total_p_s = np.abs(raw_p_s) + np.sum(np.abs(interaction_matrix), axis=1)
            total_t_s = np.abs(raw_t_s) + np.sum(np.abs(interaction_matrix), axis=0)

            # 1. Format Pixels: Upsample to Full Hardware Grid if Grouped
            total_p_s_tensor = torch.tensor(total_p_s, device=self.wrapper.device, dtype=torch.float32)
            if self.patch_group_size > 1 and "internvl" not in model_type:
                small_grid = total_p_s_tensor.view(1, 1, game.grouped_h, game.grouped_w)
                upsampled_p_s = F.interpolate(small_grid, size=(game.grid_h, game.grid_w), mode="nearest").flatten()
                pixel_shap.append(upsampled_p_s.cpu().numpy())
            else:
                pixel_shap.append(total_p_s)

            # 2. Format Tokens: Scatter into the Full Sequence Array
            t_s_full = [0.0] * seq_len
            for j, real_seq_idx in enumerate(game.valid_token_indices.cpu().numpy()):
                t_s_full[real_seq_idx] = total_t_s[j]
            token_shap.append(t_s_full)

            # ==========================================
            # STEP C: FORMAT THE SYNERGISTIC MATRIX
            # ==========================================
            # 1. Scatter the text dimension of the matrix to the full sequence length
            synergy_scattered = torch.zeros((game.n_players_image, seq_len), device=self.wrapper.device)
            for j, real_seq_idx in enumerate(game.valid_token_indices.cpu().numpy()):
                synergy_scattered[:, real_seq_idx] = torch.tensor(interaction_matrix[:, j], device=self.wrapper.device)

            # 2. Upsample the spatial dimension to the full Hardware Grid
            if self.patch_group_size > 1 and "internvl" not in model_type:
                # Reshape to (1, seq_len, grouped_h, grouped_w) for PyTorch 2D spatial interpolation
                small_synergy_grid = synergy_scattered.T.view(1, seq_len, game.grouped_h, game.grouped_w)
                upsampled_synergy = F.interpolate(small_synergy_grid, size=(game.grid_h, game.grid_w), mode="nearest")

                # Permute to requested output shape: (H, W, seq_len)
                spatial_synergy = upsampled_synergy.squeeze(0).permute(1, 2, 0)
            else:
                spatial_synergy = synergy_scattered.T.view(seq_len, game.grid_h, game.grid_w).permute(1, 2, 0)

            cross_modal_synergies.append(spatial_synergy.cpu().numpy())

        # 4. Final Format to Output Tensors
        token_shap_tensor = torch.tensor(np.array(token_shap), device=self.wrapper.device, dtype=torch.float32)

        pixel_attributions = torch.tensor(np.array(pixel_shap), device=self.wrapper.device, dtype=torch.float32)
        # pixel_attributions = self._reshape_to_spatial(pixel_attributions, model_type, inputs)

        # Stack synergies for all evaluated targets. Shape: (num_targets, H, W, seq_len)
        # synergy_attributions = torch.tensor(np.array(cross_modal_synergies),
        #                                     device=self.wrapper.device,
        #                                     dtype=torch.float32)

        # return token_shap_tensor, pixel_attributions, synergy_attributions
        return token_shap_tensor, pixel_attributions

    def _reshape_to_spatial(self, pixel_shap, model_type, inputs):
        """Maps the flat 1D pixel SHAP array into the 2D hardware grid of the specific VLM."""
        num_targets = pixel_shap.shape[0]

        if "internvl" in model_type:
            # InternVL operates on tiles, so we map the global score to the whole tile
            pixel_values = inputs["pixel_values"]
            if pixel_values.ndim == 4:
                pixel_values = pixel_values.unsqueeze(0)
            _, _, _, H, W = pixel_values.shape
            global_tile_shap = pixel_shap[:, -1].view(num_targets, 1, 1)
            return global_tile_shap.expand(-1, H, W)

        elif "qwen" in model_type:
            if "image_grid_thw" in inputs:
                grid_thw = inputs["image_grid_thw"][0].cpu().numpy().tolist()
                H, W = grid_thw[1], grid_thw[2]
                return pixel_shap.view(num_targets, H, W)
            else:
                return pixel_shap

        else:  # LLaVA & Standard ViTs
            pixel_values = inputs["pixel_values"]
            if pixel_values.ndim == 3:
                pixel_values = pixel_values.unsqueeze(0)
            _, _, H, W = pixel_values.shape
            return pixel_shap.view(num_targets, H, W)
