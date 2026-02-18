import time
import torch
import numpy as np
from collections import defaultdict

def run_xai_benchmark(
    task: str,               # e.g., "captioning", "vqa"
    model,                   # The VLM (e.g., LLaVA, Qwen)
    explainer,               # The XAI method (e.g., TAM, Chefer)
    dataloader,              # Dataset D yielding (img, prompt, masks)
    processor,               # Tokenizer/Image Processor
    ontology_mapper,         # Maps words to dataset mask categories
    faithfulness_metric,     # Your Deletion/Insertion AUC function
    plausibility_metric      # Your Pointing Game / Energy function
):
    """
    Executes a comprehensive benchmark across Dataset D for Model M and Explainer E.
    """
    # --- Global Metric Trackers ---
    dataset_results = defaultdict(list)
    
    print(f"Starting Benchmark | Task: {task} | Explainer: {explainer.__class__.__name__}")
    
    for batch_idx, sample in enumerate(dataloader):
        img = sample["image"].to(model.device)
        prompt = sample["prompt"]
        gt_masks = sample.get("category_masks", {}) # Dict of {cat_name: mask_tensor}
        
        # --- 1. TRACK COMPUTATION METRICS (START) ---
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # --- 2. GENERATION (Model M) ---
        # Get the Answer A (and cache logits/activations if needed by explainer)
        pred_results = model.predict(img, prompt, return_logits=True)
        generated_token_ids = pred_results["new_ids"].squeeze(0).tolist()
        
        # --- 3. EXPLANATION (Explainer E) ---
        # Generate attributions for the entire sequence at once (if supported)
        # heatmaps shape: (Seq_Len, H, W)
        text_attr, img_attr = explainer.explain_sequence(img, prompt, pred_results)
        
        # --- TRACK COMPUTATION METRICS (END) ---
        torch.cuda.synchronize()
        exec_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert to MB
        
        # --- 4. NLP PREPROCESSING ---
        # Group subword tokens into coherent words and max-pool their heatmaps
        words, word_indices_list = group_subwords(generated_token_ids, processor)
        
        # Filter for evaluable tokens (Nouns for Captioning, or Answer nouns for VQA)
        evaluable_words = filter_evaluable_words(words, word_indices_list, task)
        
        # --- Image-Level Trackers ---
        img_plausibility_scores = []
        img_faithfulness_scores = []
        
        # --- 5. TOKEN-LEVEL EVALUATION LOOP ---
        for word, token_indices in evaluable_words:
            
            # Pool the image attributions for this specific word
            word_img_attr = pool_heatmaps(img_attr, token_indices, method='max')
            
            # --- A. Plausibility Computation ---
            # Attempt to map the word to a ground truth visual concept
            cat_id, cat_name = ontology_mapper.map_word(word)
            
            if cat_name is not None and cat_name in gt_masks:
                gt_mask = gt_masks[cat_name]
                
                # Compute Spatial Plausibility (e.g., Pointing Game)
                p_score = plausibility_metric(gt_mask, word_img_attr)
                img_plausibility_scores.append(p_score)
            else:
                # For VQA, some answers lack bounding boxes. 
                # For Captioning, hallucinated words lack boxes.
                # We skip plausibility but STILL compute Faithfulness.
                pass
                
            # --- B. Faithfulness Computation ---
            # Compute Faithfulness (e.g., Deletion AUC) for this specific word
            # We pass the token_indices so the metric knows which probability to track
            f_score = faithfulness_metric(
                model=model,
                original_img=img,
                prompt=prompt,
                word_img_attr=word_img_attr, # The explanation to perturb by
                target_token_indices=token_indices, # The tokens to measure prob drop
                pred_results=pred_results
            )
            img_faithfulness_scores.append(f_score)

        # --- 6. AGGREGATE IMAGE-LEVEL RESULTS ---
        # If the model generated no nouns (e.g., "Yes", "No" in VQA without masks), 
        # we handle the empty lists gracefully.
        
        if img_plausibility_scores:
            dataset_results["plausibility"].append(np.mean(img_plausibility_scores))
            
        if img_faithfulness_scores:
            dataset_results["faithfulness"].append(np.mean(img_faithfulness_scores))
            
        dataset_results["exec_time"].append(exec_time)
        dataset_results["peak_vram"].append(peak_vram)
        
        # Optional: Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} | Time: {exec_time:.2f}s | VRAM: {peak_vram:.0f}MB | "
                  f"Avg Plaus: {np.mean(img_plausibility_scores) if img_plausibility_scores else 0:.2f}")

    # --- 7. FINAL DATASET AGGREGATION ---
    final_metrics = {
        "Task": task,
        "Explainer": explainer.__class__.__name__,
        "Mean_Plausibility": np.mean(dataset_results["plausibility"]) if dataset_results["plausibility"] else 0.0,
        "Mean_Faithfulness": np.mean(dataset_results["faithfulness"]) if dataset_results["faithfulness"] else 0.0,
        "Avg_Time_Per_Sample_sec": np.mean(dataset_results["exec_time"]),
        "Max_Peak_VRAM_MB": np.max(dataset_results["peak_vram"]), # VRAM is usually tracked by the absolute peak
    }
    
    return final_metrics
