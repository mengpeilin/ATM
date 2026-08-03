#!/bin/bash
# Resumable two-task schedule requested for simulator-GT RoboTwin ATM training.
set -euo pipefail

ATM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROBOTWIN_ROOT=/data/peilin/Bimanual_Manipulation/RoboTwin
PY=/data/home/peilin/miniconda3/envs/atm/bin/python
TASK_CONFIG=demo_clean
TASKS=(place_bread_skillet stack_blocks_three)
LOG_DIR=${ATM_ROOT}/logs/two_task_training

cd "$ATM_ROOT"
mkdir -p "$LOG_DIR"

wait_for_file() {
    local path=$1
    while [ ! -f "$path" ]; do sleep 30; done
}

preprocess_task() {
    local task=$1
    local data_dir="data/atm_robotwin/${TASK_CONFIG}/${task}"
    local marker="${data_dir}/.preprocess_simulator_gt_v1_done"
    if [ -f "$marker" ]; then
        $PY -m scripts.preprocess_robotwin --task "$task" --task-config "$TASK_CONFIG" \
            --train-ratio 0.95 --verify-only
        return
    fi
    if [ -e "${data_dir}/train" ] || [ -e "${data_dir}/val" ]; then
        echo "refusing stale split for $task" >&2
        exit 1
    fi
    $PY -m scripts.preprocess_robotwin --task "$task" --task-config "$TASK_CONFIG" \
        --train-ratio 0.95 --stats-only
    CUDA_VISIBLE_DEVICES=0 $PY -m scripts.preprocess_robotwin \
        --task "$task" --task-config "$TASK_CONFIG" --start-episode 0 --num-episodes 50 \
        --skip-exist True --train-ratio 0.95 > "${LOG_DIR}/prep_${task}_gpu0.log" 2>&1 &
    local p0=$!
    CUDA_VISIBLE_DEVICES=1 $PY -m scripts.preprocess_robotwin \
        --task "$task" --task-config "$TASK_CONFIG" --start-episode 50 --num-episodes 50 \
        --skip-exist True --train-ratio 0.95 > "${LOG_DIR}/prep_${task}_gpu1.log" 2>&1 &
    local p1=$!
    wait "$p0"
    wait "$p1"
    $PY -m scripts.preprocess_robotwin --task "$task" --task-config "$TASK_CONFIG" \
        --train-ratio 0.95 --verify-only
    touch "$marker"
}

train_tt() {
    local task=$1
    local short=$2
    local experiment="tt_${short}_sim_gt_1000_b256"
    $PY -m engine.train_track_transformer --config-name=robotwin_track_transformer \
        experiment="$experiment" train_gpus='[0,1]' \
        train_dataset="['./data/atm_robotwin/${TASK_CONFIG}/${task}/train']" \
        val_dataset="['./data/atm_robotwin/${TASK_CONFIG}/${task}/val']" \
        epochs=1000 batch_size=256 val_batch_size=256 gradient_accumulate_steps=2 \
        > "${LOG_DIR}/tt_${task}.log" 2>&1
    local result_dir
    result_dir=$(ls -td "${ATM_ROOT}/results/track_transformer/${experiment}_"* | head -1)
    test -f "${result_dir}/model_best.ckpt"
    printf '%s\n' "$result_dir" > "${LOG_DIR}/tt_${task}.result_dir"
}

train_policy() {
    local task=$1
    local short=$2
    local gpu=$3
    local tt_dir
    tt_dir=$(cat "${LOG_DIR}/tt_${task}.result_dir")
    local experiment="atm_dp_${short}_sim_gt_tt1000"
    CUDA_VISIBLE_DEVICES=$gpu $PY diffusion_policy/train.py \
        --config-name=train_robotwin_atm_workspace \
        experiment="$experiment" \
        task.dataset.train_dataset_dirs="['./data/atm_robotwin/${TASK_CONFIG}/${task}/train']" \
        task.dataset.val_dataset_dirs="['./data/atm_robotwin/${TASK_CONFIG}/${task}/val']" \
        policy.track_fn="$tt_dir" training.checkpoint_every=100 \
        > "${LOG_DIR}/policy_${task}.log" 2>&1
    local result_dir
    result_dir=$(ls -td "${ATM_ROOT}/results/policy/${experiment}_"* | head -1)
    test -f "${result_dir}/checkpoints/last.ckpt"
    for epoch in 100 200 300 400 500; do
        test -f "${result_dir}/checkpoints/epoch_${epoch}.ckpt"
    done
    printf '%s\n' "$result_dir" > "${LOG_DIR}/policy_${task}.result_dir"
}

evaluate_policy() {
    local task=$1
    local gpu=$2
    local policy_dir
    policy_dir=$(cat "${LOG_DIR}/policy_${task}.result_dir")
    cd "$ROBOTWIN_ROOT"
    bash policy/ATM_DP/eval.sh "$task" "$TASK_CONFIG" "$TASK_CONFIG" 0 "$gpu" \
        "${policy_dir}/checkpoints/last.ckpt" \
        "${ATM_ROOT}/data/atm_robotwin/${TASK_CONFIG}/${task}/norm_stats.json" 50 \
        > "${LOG_DIR}/eval_${task}.log" 2>&1
    cd "$ATM_ROOT"
}

# The first preprocessing job was already launched interactively; wait for its verified marker.
wait_for_file "data/atm_robotwin/${TASK_CONFIG}/place_bread_skillet/.preprocess_simulator_gt_v1_done"
preprocess_task stack_blocks_three
$PY -m scripts.split_libero_dataset --folder ./data/atm_robotwin/ --train_ratio 0.95

# Each transformer owns both GPUs, so these must remain sequential.
train_tt place_bread_skillet bread_skillet
train_tt stack_blocks_three blocks_three

# Each diffusion policy owns one GPU; train them concurrently only after both transformers finish.
train_policy place_bread_skillet bread_skillet 0 &
policy0=$!
train_policy stack_blocks_three blocks_three 1 &
policy1=$!
wait "$policy0"
wait "$policy1"

evaluate_policy place_bread_skillet 0 &
eval0=$!
evaluate_policy stack_blocks_three 1 &
eval1=$!
wait "$eval0"
wait "$eval1"
touch "${LOG_DIR}/.complete"
