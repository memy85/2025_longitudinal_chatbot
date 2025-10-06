#!/bin/bash

CUDA_VISIBLE_DEVICES=2
# TORCH_CUDA_ARCH_LIST=7.5
CHATBOT_PATH=/data/data_user/public_models/chatbot_noteaid/checkpoint-4/

# launch
vllm serve $CHATBOT_PATH \
	--port 18001 \
	--tensor-parallel-size 1 \
	--max-model-len 8192 \
	--dtype float32 
	# --gpu-memory-utilization 0.9 \
