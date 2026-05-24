#!/bin/bash
# start_in_tmux.sh

# tmux a -t 1
CUDA_VISIBLE_DEVICES=0 sglang serve \
     --model-path ./models/Qwen3-4B --port 30000 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85 --reasoning-parser qwen3

# tmux a -t 2
CUDA_VISIBLE_DEVICES=1 sglang serve \
     --model-path ./models/Qwen3-8B --port 30001 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85 --reasoning-parser qwen3
# tmux a -t 3
CUDA_VISIBLE_DEVICES=2 sglang serve \
     --model-path ./models/Qwen3-14B --port 30002 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85 --reasoning-parser qwen3
# tmux a -t 4
CUDA_VISIBLE_DEVICES=3,4 sglang serve \
     --model-path ./models/Qwen3-32B --tp-size 2 \
     --port 30003 --host 0.0.0.0 \
     --context-length 32768 --mem-fraction-static 0.85
# tmux a -t 6
CUDA_VISIBLE_DEVICES=5 sglang serve \
     --model-path ./models/Llama-3.1-8B-Instruct --port 30004 --host 0.0.0.0
# tmux a -t 7
CUDA_VISIBLE_DEVICES=0,5,6,7 sglang serve \
     --model-path ./models/Llama-3.1-70B-Instruct --tp-size 4 \
     --port 30005 --host 0.0.0.0