#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7
TORCH_CUDA_ARCH_LIST=7.5
CHATBOT_PATH=/data/data_user/public_models/DeepSeek-R1-Distill-Llama-8B/

# launch
vllm serve $CHATBOT_PATH \
	--port 18000 \
	--tensor-parallel-size 2 \
	--gpu-memory-utilization 0.9 \
	--dtype float16 \
	--max-model-len 8192
