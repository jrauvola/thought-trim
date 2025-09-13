#!/bin/bash
VLLM_WORKER_MULTIPROC_METHOD=spawn python3 ./thought-anchors/generate_rollouts.py --provider Local --num_problems 1 --num_rollouts 10 --batch_size 1 --repetition_penalty 1.1 --model deepseek/DeepSeek-R1-Distill-Qwen-1.5B --output_dir qwen_3_1_7b_9_12_2025_1054pm_repetition_1_1_law_2 -mmlu dataset_mmlu/mmlu_pro_sections/biology_sample_100.json
