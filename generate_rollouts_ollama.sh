#!/bin/bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python3 ./thought-anchors/generate_rollouts.py --provider Local --num_problems 1 --num_rollouts 100 --batch_size 100 --model deepseek/deepseek-r1-distill-qwen-1.5b --output_dir test_100_rollouts_fix_2/ --repetition_penalty 1.1
