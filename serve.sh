#!/bin/bash
# start_in_tmux.sh


CUDA_VISIBLE_DEVICES=0 sglang serve \
     --model-path ./models/Qwen3-4B --port 30000 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85 --reasoning-parser qwen3


CUDA_VISIBLE_DEVICES=1 sglang serve \
     --model-path ./models/Qwen3-8B --port 30001 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85 --reasoning-parser qwen3"

CUDA_VISIBLE_DEVICES=2 sglang serve \
     --model-path ./models/Qwen3-14B --port 30002 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85 --reasoning-parser qwen3"

CUDA_VISIBLE_DEVICES=3,4 sglang serve \
     --model-path ./models/Qwen2.5-32B-Instruct --tp-size 2 \
     --port 30003 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85"

