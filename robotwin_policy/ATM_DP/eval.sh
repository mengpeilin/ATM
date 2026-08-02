#!/bin/bash
set -euo pipefail

task_name=${1:?usage: eval.sh TASK TASK_CONFIG CKPT_SETTING SEED GPU CKPT NORM_STATS [TEST_NUM]}
task_config=${2:?}
ckpt_setting=${3:?}
seed=${4:?}
gpu_id=${5:?}
ckpt_path=${6:?}
norm_stats_path=${7:?}
test_num=${8:-100}

export CUDA_VISIBLE_DEVICES=$gpu_id
python script/eval_policy.py --config policy/ATM_DP/deploy_policy.yml \
    --overrides \
    --task_name "$task_name" \
    --task_config "$task_config" \
    --ckpt_setting "$ckpt_setting" \
    --seed "$seed" \
    --policy_name ATM_DP \
    --test_num "$test_num" \
    --ckpt_path "$ckpt_path" \
    --norm_stats_path "$norm_stats_path"
