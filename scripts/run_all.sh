#!/bin/bash

# Stop the script immediately if any command fails
set -e 

echo "========================================="
echo "   Starting VLM XAI Benchmark Suite      "
echo "========================================="

# 1. Define your configurations
# Add or remove paths here to change what the script runs
MODELS=(
    #"configs/models/llava.yaml"
    "configs/models/qwenvl.yaml"
    #"configs/models/internvl.yaml"
)

DATASETS=(
    "configs/datasets/repope.yaml"
)

# Pass all explainers as a single string separated by spaces
# EXPLAINERS="random rollout llavacam lxt tam"
# EXPLAINERS="tam gradcam gradxrollout integratedgradients"
EXPLAINERS="random inputxgradients llavacam rollout lxt"

GPU_ID=1

# 2. Nested loops to run every combination
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        echo ""
        echo ">>> Launching Job: Model=$MODEL | Dataset=$DATASET"
        echo ">>> Explainers: $EXPLAINERS"
        
        # 3. Execute the Python script
        # Notice we omitted --max_samples so it runs the whole dataset!
        python -m src.benchmark \
            --model_config "$MODEL" \
            --dataset_config "$DATASET" \
            --explainers $EXPLAINERS \
            --gpu_id $GPU_ID \
            # --max_samples 200  <-- Uncomment this line for a quick test run!
            
        echo "<<< Job Finished. Moving to next configuration..."
        sleep 5 # Give the OS a few seconds to fully flush the GPU VRAM
        
    done
done

echo "========================================="
echo "   All Benchmarks Completed Successfully! "
echo "========================================="