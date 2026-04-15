import argparse
import os
import time
import traceback
import json

import torch
import yaml
from tqdm import tqdm

# --- ABSTRACTED FACTORIES & UTILS ---
from src.datasets.factory import get_dataloader 
from src.explainers.utils import get_decision_token_index, save_to_jsonl
from src.metrics import FaithfulnessMetric
from src.models.factory import load_vlm

# Import the base Oracle
from src.explainers import OracleExplainer, MismatchedExplainer



def load_yaml(file_path):
    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_experiment_1_2(args):
    # 1. Load Configurations
    dataset_config = load_yaml(args.dataset_config)
    model_config = load_yaml(args.model_config)

    # Setup Output Directory specifically for the Mismatch test
    output_dir = os.path.join(
        args.output_dir, model_config["name"], f"{dataset_config['name']}_mismatch"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Model 
    print(f"[*] Loading Model: {model_config['name']}...")
    model_wrapper = load_vlm(
        model_config=model_config,
        attn_implementation=None, 
        gpu_node=args.gpu_id,
        output_attentions=False,
    )

    # 3. Load the ORACLE Dataset
    dl = get_dataloader(dataset_config)

    # 4. Initialize Metrics
    pert_steps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    tok = model_wrapper.processor.tokenizer
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    faith_metrics = FaithfulnessMetric(
        perturbation_steps=pert_steps,
        pad_token_id=pad_token_id,
        special_token_ids=model_wrapper.special_token_ids,
        mask_value=0.0,
        filter_keywords=True,
    )

    # 5. Initialize the Explainers to Compare
    # We test both so we can prove the Delta between a true Oracle and a Mismatched one
    explainers_to_test = {
        "Oracle": OracleExplainer(model_wrapper),
        "Mismatched-Explainer": MismatchedExplainer(model_wrapper)
    }

    # ---------------------------------------------------------
    # OUTER LOOP: Iterate over the Explainers
    # ---------------------------------------------------------
    for explainer_name, explainer in explainers_to_test.items():
        try:
            print(f"\n{'=' * 50}\n[*] Evaluating: {explainer_name} \n{'=' * 50}")

            run_name = f"{model_config['name']}_exp_1_2_{explainer_name}"
            output_file = os.path.join(output_dir, f"{run_name}_results.jsonl")

            # --- RESUME LOGIC ---
            processed_indices = set()
            if os.path.exists(output_file):
                print(f"[*] Found existing results. Scanning for completed samples...")
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "sample_idx" in data:
                                    processed_indices.add(data["sample_idx"])
                            except json.JSONDecodeError:
                                pass
                print(f"[*] Skipping {len(processed_indices)} already processed samples.")

            # ---------------------------------------------------------
            # INNER LOOP: Evaluate Dataset Samples
            # ---------------------------------------------------------
            valid_samples_evaluated = 0
            
            for idx, sample in enumerate(tqdm(dl, desc=f"Evaluating {explainer_name}")):
                if args.max_samples is not None and valid_samples_evaluated >= args.max_samples:
                    print(f"[*] Reached target of {args.max_samples} valid samples. Stopping.")
                    break

                if idx in processed_indices:
                    # If we already processed it, count it towards our valid total
                    valid_samples_evaluated += 1
                    continue 

                # ==============================================================
                # THE DISTRACTOR CHECK (Skip images with no second object)
                # ==============================================================
                dist_cat = sample.get("distractor_category")
                # Handle PyTorch DataLoader batching (if it returns a list of strings)
                if isinstance(dist_cat, list):
                    dist_cat = dist_cat[0]
                    
                if dist_cat == "background":
                    # Skip this image completely because it doesn't have an intra-image distractor
                    continue
                # ==============================================================

                img = sample["image"]
                question = sample["question"]
                image_id = sample.get("image_id", f"unknown_{idx}")
                
                # Extract Masks
                keyword = sample.get("object_name")
                oracle_mask_2d = sample.get("pixel_oracle_mask")
                mismatched_mask_2d = sample.get("distractor_mask") 

                try:
                    # 1. Forward Pass
                    inputs = model_wrapper.get_inputs(img, question)
                    pred_results = model_wrapper.predict(inputs, return_logits=False)

                    # 2. Identify the Decision Token
                    # yes_no_tok_idx = get_decision_token_index(
                    #     pred_results["new_ids"],
                    #     text_answer=pred_results["text"],
                    #     tokenizer=tok,
                    # )
                    # if yes_no_tok_idx is None:
                    #     yes_no_tok_idx = 0
                    yes_no_tok_idx = 0

                    # 3. Generate Attributions
                    start_time = time.perf_counter()
                    
                    # We pass BOTH masks in the kwargs. The specific explainer class 
                    # will automatically grab the one it needs!
                    text_attrs, img_attrs = explainer.attribute(
                        img,
                        text=question,
                        target_indices=yes_no_tok_idx,
                        pred_results=pred_results,
                        keyword=keyword,               
                        oracle_mask_2d=oracle_mask_2d,       # Used by standard Oracle
                        mismatched_mask_2d=mismatched_mask_2d # Used by MismatchedExplainer
                    )
                    xai_gen_time = time.perf_counter() - start_time

                    # 4. Prepare XAI Results
                    xai_result = {
                        "inputs": inputs,
                        "target_ids": pred_results["new_ids"].unsqueeze(0),
                        "pixel_attribution": img_attrs[0:1], 
                        "token_attribution": text_attrs[0:1],
                    }

                    faith_sample = {"image": img, "text": question}

                    # 5. Compute Metrics
                    faith_scores = faith_metrics.compute(
                        model_wrapper, faith_sample, xai_result
                    )

                    # 6. Logging
                    log_dict = {
                        "sample_idx": idx,
                        "image_id": image_id,
                        "explainer": explainer_name,
                        "question": question,
                        "label": sample.get("label", "yes"),
                        "prediction": pred_results.get("text"),
                        "distractor_category": dist_cat, # Log what we swapped it with!
                        "xai_gen_time": xai_gen_time,
                    }
                    log_dict.update(faith_scores)

                    save_to_jsonl(log_dict, output_file)
                    
                    # Successfully evaluated a valid sample!
                    valid_samples_evaluated += 1

                    del pred_results, img_attrs, text_attrs
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"[!] {explainer_name} failed on sample {idx}: {e}")
                    traceback.print_exc()
                    continue

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[!] ERROR: '{explainer_name}' crashed completely!")
            print(f"[!] Details: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1.2: Cross-Modal Illusion")

    parser.add_argument("--model_config", type=str, required=True, help="Path to model YAML")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset YAML (Must be Oracle dataset)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU node to use")
    
    parser.add_argument("--max_samples", type=int, default=200, help="Max VALID samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="logs/experiment_1", help="Where to save logs")

    args = parser.parse_args()

    try:
        run_experiment_1_2(args)
    except Exception as e:
        print(f"[FATAL] Benchmark crashed: {e}")

