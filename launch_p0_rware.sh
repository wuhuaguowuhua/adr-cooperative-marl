#!/usr/bin/env bash
# Reproduce the four-arm SEAC + RWARE controlled schedule comparison.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-adr-p0-rware}"
SEEDS="${SEEDS:-42 123 456}"
ARMS="${ARMS:-baseline static linear adr}"

run_one() {
  local arm="$1"
  local seed="$2"
  local group
  local -a snd

  case "$arm" in
    baseline)
      group="Baseline"
      snd=(SND_ENABLE=0)
      ;;
    static)
      group="Static"
      snd=(
        SND_ENABLE=1 SND_LOSS_MODE=rcdc SND_TRIGGER=static SND_METRIC=l2
        SND_DIVERSITY_COEF=0.1 SND_ETA_MAX=0.10 SND_WARMUP_RATIO=0.0
      )
      ;;
    linear)
      group="Linear"
      snd=(
        SND_ENABLE=1 SND_LOSS_MODE=rcdc SND_TRIGGER=linear SND_METRIC=l2
        SND_DIVERSITY_COEF=0.1 SND_ETA_MAX=0.10 SND_WARMUP_RATIO=0.0
      )
      ;;
    adr)
      group="ADR"
      snd=(
        SND_ENABLE=1 SND_LOSS_MODE=rcdc SND_TRIGGER=feedback SND_METRIC=l2
        SND_DIVERSITY_COEF=0.1 SND_ETA_MAX=0.10
        SND_WARMUP_RATIO=0.0375
        SND_PROACTIVE_RAMP_END=0.225 SND_PROACTIVE_RAMP_POWER=2.0
        SND_PROACTIVE_TIME_SHUTOFF=1.0 SND_DIVERSITY_CEILING=1.0
        SND_FEEDBACK_GAIN=0.05 SND_FEEDBACK_TREND_THRESH=0.001
        SND_FEEDBACK_INIT_DIRECTION=1.0 SND_FEEDBACK_DECEL_THRESH=0.5
        SND_FEEDBACK_MIN_PEAK=1.0 SND_FEEDBACK_MIN_INTERVAL=50
        SND_FEEDBACK_MIN_PROGRESS=0.40
      )
      ;;
    *)
      echo "Unknown arm: $arm" >&2
      exit 2
      ;;
  esac

  echo "Running SEAC+RWARE arm=${arm} seed=${seed}"
  local -a command=(
    env
    ALGO=a2c
    "WANDB_PROJECT=$WANDB_PROJECT"
    "WANDB_RUN_GROUP=$group"
    "WANDB_RUN_NAME=SEAC-${arm}-s${seed}"
    "WANDB_DISABLED=${WANDB_DISABLED:-false}"
    SEAC_DISABLE_VIDEO=1
    "${snd[@]}"
    "$PYTHON_BIN" train.py with paper_rware_seac
    "algorithm.seed=$seed"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

for arm in $ARMS; do
  for seed in $SEEDS; do
    run_one "$arm" "$seed"
  done
done
