import os
import time
import argparse
import yaml
from tqdm import tqdm
import torch
import wandb

# --- ABSTRACTED FACTORIES & UTILS ---
from src.datasets.factory import get_dataloader
from src.models.factory import load_vlm
from src.explainers.factory import get_explainer
from src.metrics import FaithfulnessMetric
from src.explainers.utils import save_to_jsonl, get_decision_token_index

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_benchmark(args):
    # 1. Load Configurations
    dataset_config = load_yaml(args.dataset_config)
    model_config = load_yaml(args.model_config)
    
    # --- AUTO-PATH RESOLUTION ---
    explainer_paths = []
    for exp in args.explainers:
        if exp.endswith(".yaml"):
            explainer_paths.append(exp) # User provided a direct path
        else:
            explainer_paths.append(f"configs/explainers/{exp}.yaml") # Auto-resolve!
            
    # Load them using the resolved paths
    explainer_configs = [load_yaml(path) for path in explainer_paths]
    
    # Setup Output Directory
    output_dir = os.path.join(args.output_dir, model_config["name"], dataset_config["name"])
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Attention "Lookahead" Optimization
    needs_attention = any(cfg.get("requires_attention", False) for cfg in explainer_configs)
    attn_mode = "eager" if needs_attention else None
    print(f"[*] Attention Implementation set to: {attn_mode}")

    # 3. Load Model (ONCE per script execution)
    model_wrapper = load_vlm(
        model_config=model_config,
        attn_implementation=attn_mode,
        gpu_node=args.gpu_id,
        output_attentions=needs_attention,
    )
    # Store the config on the wrapper so LXT Explainer can reload the model if needed!
    model_wrapper.model_config = model_config 
    
    # 4. Load Dataset
    dl = get_dataloader(dataset_config)
    
    # 5. Initialize Metrics
    pert_steps = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    tok = model_wrapper.processor.tokenizer
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    faith_metrics = FaithfulnessMetric(
        perturbation_steps=pert_steps,
        pad_token_id=pad_token_id,
        special_token_ids=model_wrapper.special_token_ids,
        mask_value=0.0,
        filter_keywords=False
    )

    # ---------------------------------------------------------
    # OUTER LOOP: Iterate over Requested Explainers
    # ---------------------------------------------------------
    for explainer_path in explainer_paths:
        # Load explainer dynamically, injecting model config for CAM layers
        explainer, explainer_name = get_explainer(explainer_path, model_wrapper, model_config)

        try:
            
            print(f"\n{'='*50}\n[*] Evaluating: {explainer_name} on {model_config['name']}\n{'='*50}")
            
            # Initialize Weights & Biases
            run_name = f"{model_config['name']}_{dataset_config['name']}_{explainer_name}"
            # wandb.init(
            #     project="vlm-xai-benchmark", 
            #     name=run_name,
            #     config={
            #         "model": model_config,
            #         "dataset": dataset_config,
            #         "explainer": load_yaml(explainer_path),
            #         "max_samples": args.max_samples
            #     }
            # )

            output_file = os.path.join(output_dir, f"{run_name}_results.jsonl")

            # ---------------------------------------------------------
            # INNER LOOP: Evaluate Dataset Samples
            # ---------------------------------------------------------
            for idx, sample in enumerate(tqdm(dl, desc=f"Evaluating {explainer_name}")):
                if args.max_samples is not None and idx >= args.max_samples:
                    break
                
                img = sample["image"]
                question = sample["question"]
                
                try:
                    # 1. Forward Pass
                    inputs = model_wrapper.get_inputs(img, question)
                    pred_results = model_wrapper.predict(inputs, return_logits=True)

                    # 2. Identify the Decision Token (Yes/No)
                    yes_no_tok_idx = get_decision_token_index(
                        pred_results["new_ids"],
                        text_answer=pred_results["text"],
                        tokenizer=tok
                    )
                    if yes_no_tok_idx is None:
                        yes_no_tok_idx = 0 # Fallback
                    
                    # 3. Generate Attributions (Targeted)
                    start_time = time.perf_counter()
                    text_attrs, img_attrs = explainer.attribute(
                        img, text=question, target_indices=yes_no_tok_idx, pred_results=pred_results
                    )
                    xai_gen_time = time.perf_counter() - start_time

                    # BUG FIX: Explainer returned only the targeted rows, so we slice 0:1
                    xai_result = {
                        "inputs": inputs,
                        "target_ids": pred_results["new_ids"].unsqueeze(0),
                        "pixel_attribution": img_attrs[0:1],
                        "token_attribution": text_attrs[0:1]
                    }

                    # 4. Compute Metrics
                    faith_scores = faith_metrics.compute(model_wrapper, {}, xai_result)

                    # 5. Logging
                    log_dict = {
                        "sample_idx": idx,
                        "image_id": sample.get("image_id", f"unknown_{idx}"),
                        "explainer": explainer_name,
                        "question": question,
                        "label": sample.get("label"),
                        "prediction": pred_results.get("text"),
                        "xai_gen_time": xai_gen_time
                    }
                    log_dict.update(faith_scores)
                    
                    # wandb.log(log_dict, step=idx)
                    save_to_jsonl(log_dict, output_file)

                except Exception as e:
                    print(f"[!] Explainer failed on sample {idx}: {e}")
                    continue
                
            # Cleanup before moving to the next explainer
            print(f"[*] Finished {explainer_name}. Cleaning up GPU memory...")
            if hasattr(explainer, "cleanup"):
                explainer.cleanup()
            del explainer
            torch.cuda.empty_cache()
            # wandb.finish()

        # --- THE NEW CRASH CATCHER ---
        except Exception as e:
            print(f"\n[!] ERROR: Explainer '{explainer_path}' crashed completely!")
            print(f"[!] Exception Details: {e}")
            print(f"[!] Skipping this explainer and moving to the next one...\n")
            
            # 1. Safely force-delete the explainer from VRAM if it partially initialized
            if explainer is not None:
                if hasattr(explainer, "cleanup"):
                    try:
                        explainer.cleanup()
                    except:
                        pass
                del explainer
            torch.cuda.empty_cache()
            
            # 2. Tell W&B that this specific run crashed, so it doesn't hang
            # if wandb.run is not None:
            #     wandb.finish(exit_code=1)
            
            # 3. Move on to the next explainer!
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM XAI Benchmark")
    
    # CLI Arguments now just point to YAML files!
    parser.add_argument("--model_config", type=str, required=True, help="Path to model YAML")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset YAML")
    parser.add_argument("--explainers", nargs="+", required=True, help="List of paths to explainer YAMLs")
    
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU node to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="logs/results_final", help="Where to save logs")
    
    args = parser.parse_args()
    
    try:
        run_benchmark(args)
    except Exception as e:
        print(f"[FATAL] Benchmark crashed: {e}")
        # wandb.finish(exit_code=1)
