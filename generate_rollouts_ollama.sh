#!/bin/bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python3 ./thought-anchors/generate_rollouts.py --provider Local --num_problems 1 --num_rollouts 10 --batch_size 1 --model Qwen/Qwen3-1.7B --output_dir qwen_3_1_7b_100_2 --repetition_penalty 1.1 -mmlu dataset_mmlu/mmlu_pro_sections/biology_sample_100.json
