#!/bin/bash
# End-to-end ATM-DP pipeline for one RoboTwin task: preprocess -> split -> track transformer -> DP.
#
#   bash scripts/run_task_pipeline.sh <task_name> [task_config] [preprocess_gpus] [train_gpus]
#
# Example:
#   bash scripts/run_task_pipeline.sh place_dual_shoes demo_clean "0,1" "[0,1]"
#
# Track Transformer hyperparameters follow ATM. Stage 2 uses RoboTwin's image DP settings with a
# 16-step horizon, eight executed actions and eight data workers.
set -euo pipefail

TASK=${1:?usage: run_task_pipeline.sh <task_name> [task_config] [preprocess_gpus] [train_gpus]}
TASK_CONFIG=${2:-demo_clean}
PREP_GPUS=${3:-0,1}
TRAIN_GPUS=${4:-[0,1]}
STAGE2_GPU=${PREP_GPUS%%,*}

ATM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=/data/home/peilin/miniconda3/envs/atm/bin/python
DATA_DIR=./data/atm_robotwin/${TASK_CONFIG}/${TASK}
LOG_DIR=${ATM_ROOT}/logs
SHORT=$(echo "$TASK" | sed 's/^place_//; s/^stack_//')

cd "$ATM_ROOT"
mkdir -p "$LOG_DIR"

echo "=== [$TASK] 1/4 preprocess ==="
if [ -f "${DATA_DIR}/.preprocess_native_rgb_done" ]; then
    echo "already preprocessed, skipping"
else
    # Compute train-only statistics once before workers start, avoiding validation leakage and races.
    $PY -m scripts.preprocess_robotwin \
        --task "$TASK" --task-config "$TASK_CONFIG" --train-ratio 0.95 --stats-only

    # Shard the episodes across the available GPUs; CoTracker dominates the runtime here.
    IFS=',' read -ra GPU_ARR <<< "$PREP_GPUS"
    NGPU=${#GPU_ARR[@]}
    NEP=$(ls "/data/peilin/Bimanual_Manipulation/RoboTwin/data/${TASK}/${TASK_CONFIG}/data/" | grep -c 'episode.*\.hdf5')
    CHUNK=$(( (NEP + NGPU - 1) / NGPU ))
    pids=()
    for i in "${!GPU_ARR[@]}"; do
        START=$(( i * CHUNK ))
        CUDA_VISIBLE_DEVICES=${GPU_ARR[$i]} $PY -m scripts.preprocess_robotwin \
            --task "$TASK" --task-config "$TASK_CONFIG" \
            --start-episode $START --num-episodes $CHUNK --skip-exist True \
            > "${LOG_DIR}/prep_${TASK}_gpu${GPU_ARR[$i]}.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    touch "${DATA_DIR}/.preprocess_native_rgb_done"
fi

echo "=== [$TASK] 2/4 train/val split ==="
# Idempotent: split_libero_dataset skips any task that already has a train/ folder.
$PY -m scripts.split_libero_dataset --folder ./data/atm_robotwin/ --train_ratio 0.95

echo "=== [$TASK] 3/4 track transformer ==="
$PY -m engine.train_track_transformer --config-name=robotwin_track_transformer \
    experiment="tt_${SHORT}" train_gpus="$TRAIN_GPUS" \
    train_dataset="['${DATA_DIR}/train']" val_dataset="['${DATA_DIR}/val']" \
    > "${LOG_DIR}/tt_${TASK}.log" 2>&1

TT_DIR=$(ls -td ${ATM_ROOT}/results/track_transformer/tt_${SHORT}_* | head -1)
echo "track transformer: $TT_DIR"
test -f "${TT_DIR}/model_best.ckpt" || { echo "ERROR: no model_best.ckpt in $TT_DIR"; exit 1; }

echo "=== [$TASK] 4/4 ATM diffusion policy ==="
CUDA_VISIBLE_DEVICES=$STAGE2_GPU $PY diffusion_policy/train.py \
    --config-name=train_robotwin_atm_workspace \
    experiment="atm_dp_${SHORT}" \
    task.dataset.dataset_dirs="['${DATA_DIR}']" \
    policy.track_fn="$TT_DIR" \
    > "${LOG_DIR}/policy_${TASK}.log" 2>&1

POLICY_DIR=$(ls -td ${ATM_ROOT}/results/policy/atm_dp_${SHORT}_* | head -1)
echo "=== [$TASK] DONE ==="
echo "policy: $POLICY_DIR"
echo "install robotwin_policy/ATM_DP under RoboTwin/policy, then evaluate its best checkpoint:"
echo "  bash policy/ATM_DP/eval.sh $TASK $TASK_CONFIG $TASK_CONFIG 0 0 \\"
echo "    ${POLICY_DIR}/checkpoints/best.ckpt ${ATM_ROOT}/${DATA_DIR#./}/norm_stats.json"
