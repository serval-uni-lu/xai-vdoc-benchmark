#!/bin/bash

# Stop the script immediately if any command fails
set -e 

echo "========================================="
echo "   Starting VLM XAI Benchmark Suite      "
echo "========================================="

# 1. Define your configurations
# Add or remove paths here to change what the script runs
MODELS=(
    "configs/models/llava.yaml"
    #"configs/models/qwenvl.yaml"
    #configs/models/internvl.yaml"
)

DATASETS=(
    "configs/datasets/repope.yaml"
    #"configs/datasets/repope_oracle.yaml"
    #"configs/datasets/coco.yaml"
)

OUTPUT_DIR=(
    "logs/results"
)

# Pass all explainers as a single string separated by spaces
#EXPLAINERS="gradcam integratedgradients lxt"
#EXPLAINERS="tam inputxgradients gradxrollout " 
EXPLAINERS="random llavacam rollout"

# EXPLAINERS="tam integratedgradients llavacam lxt"
# EXPLAINERS="random rollout inputxgradients gradxrollout gradcam tam integratedgradients llavacam lxt"
# EXPLAINERS="tam"

GPU_ID=1

# 2. Nested loops to run every combination
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        
        echo ""
        echo ">>> Launching Job: Model=$MODEL | Dataset=$DATASET"
        echo ">>> Explainers: $EXPLAINERS"
        
        # 3. Execute the Python script
        # Notice we omitted --max_samples so it runs the whole dataset!
        #python -m src.benchmarks.repope \
        python -m src.benchmarks.repope \
            --model_config "$MODEL" \
            --dataset_config "$DATASET" \
            --gpu_id $GPU_ID \
            --output_dir $OUTPUT_DIR \
            --explainers $EXPLAINERS \
            # --max_samples 200 \
            
            
        echo "<<< Job Finished. Moving to next configuration..."
        sleep 5 # Give the OS a few seconds to fully flush the GPU VRAM
        
    done
done

echo "========================================="
echo "   All Benchmarks Completed Successfully! "
echo "========================================="
