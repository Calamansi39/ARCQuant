#!/bin/bash
cd /gemini/code/NMSparsity/ARC_lbc
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/gemini/code/NMSparsity/lm-evaluation-harness CUDA_VISIBLE_DEVICES=0 /gemini/code/envs/smoothquant/bin/python model/main.py \
  /gemini/code/checkpoints/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
  --act_sort_metric max \
  --dataset wikitext2 \
  --tasks arc_challenge,arc_easy,boolq,openbookqa,piqa,rte,winogrande \
  --lm_eval_num_fewshot 0 \
  --lm_eval_batch_size 16 \
  --eval_devices cuda:0 \
  --quant_type NVFP4 \
  --sparse_method rsparse \
  --sparsity 0.5 \
  --output_json /gemini/code/NMSparsity/ARC_lbc/results/rsparse_bench_20260318/rest.json \
  > /gemini/code/NMSparsity/ARC_lbc/results/rsparse_bench_20260318/rest.log 2>&1
