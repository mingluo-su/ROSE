#!/bin/bash

# 设置 Hugging Face 镜像地址
export HF_ENDPOINT=https://hf-mirror.com

# 最简单的实验函数
run_experiment() {
    local model_path="$1"
    local gpu_ids="$2"
    local method="$3"
    
    export CUDA_VISIBLE_DEVICES="$gpu_ids"
    
    local model_basename=$(basename "$model_path")
    local experiment_name="${method}_${model_basename}"
    
    echo "运行: $method 剪枝 - 模型: $model_basename - GPU: $gpu_ids"
    
    python main.py \
        --model_path "$model_path" \
        --sparsity_type 2:4 \
        --sparsity_ratio 0.5 \
        --prune_method "$method" \
        --eval_zero_shot \
        --distribute \
       
}

# 使用示例
run_experiment "/home/sumingluo/models/llama2-7b" "5,7" "wanda" 
# run_experiment "/home/sumingluo/models/qwen3-32b" "0,1,2,3,4,5,6,7"  "dense" &
# run_experiment "/home/sumingluo/models/qwen3-0.6b" "6,7" "wanda" &
# run_experiment "/home/sumingluo/models/qwen3-0.6b" "2,3" "dense" &
