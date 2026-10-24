#!/bin/bash -l
# Few-shot RoMa retinal training sweep over the ablation arms.
#
# One cross-domain training run per variant. Every arm has to be *trained* under
# its own configuration as well as scored under it: a training-time augmentation and a
# module that co-adapts with the decoder cannot be switched on after the fact without
# measuring something other than the arm one means to report.
#
# The arms (see experiments/utils/variants.py for the table):
#   zero                     off-the-shelf RoMa, no fine-tuning -- nothing to train, skipped
#   baseline                 plain RoMa fine-tuning, no adaptation at all (the control)
#   few_shot                 support-conditioned ridge residual only
#   few_shot+perturb         + im_B channel-stat perturbation
#   few_shot+freq            + frequency modulation
#   few_shot+perturb+freq    both, the target configuration
#   gamma0+perturb           perturbation without the support residual   } counterfactuals,
#   gamma0+freq              frequency modulation without the support residual } off the
#   gamma0+perturb+freq      both, without the support residual           } default sweep
#
# The gamma0 family is the counterfactual for the few-shot claim: same components, but
# the ridge residual is never built, so the support prior is dead for the whole run
# rather than amputated after training. Plain gamma0 -- neither component -- is not an
# arm; it is `baseline` with dead weights. These arms stay defined in
# experiments/utils/variants.py; name them in VARIANTS to train them.
#
# Every arm is 1-shot: one support pair conditions each episode, at train time and at
# test time alike.
#
# Train on FIRE with --fold none, using FIRE/{train,val}_list.txt. The best checkpoint
# is selected on FIRE val only. Target datasets are not seen during training.
#
# Checkpoint layout -- the variant is part of the path, so two arms can never write over
# each other:
#   ${CKPT_DIR}/${GROUP}/<variant>/<subset>/foldnone/roma_fewshot_<subset>_{best,latest}.pth
#
# Re-running the sweep resumes it. Each arm's step count is read from the progress file
# beside its checkpoints before Python is launched, so an arm already at its target is
# skipped outright rather than paying for a model build to discover it is done, and one
# that stopped short -- an OOM, say -- continues from _latest.pth with its optimizer, LR
# schedule, AMP loss scale and best-so-far intact. Raising the step count trains an arm
# further; deleting its fold directory starts it over.
#
# The defaults below encode the current recipe (medium resolution, multi-scale
# alignment, 8 loader workers). Every
# one stays overridable from the command line, VARIANTS included:
#   VARIANTS="baseline few_shot+perturb+freq" bash script/train_roma_fewshot_retina.sh
# Run in a prepared GPU environment; override DATA_ROOT for another machine.
# PYRUN="uv run python" can be used instead of the active environment's Python.

SCRIPTPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPTPATH}/.." && pwd)"
cd "${PROJECT_DIR}" || exit 1

PYRUN="${PYRUN:-python}"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TORCH_HOME}/mpl}"

VARIANTS="${VARIANTS:-few_shot+perturb+freq few_shot few_shot+perturb few_shot+freq gamma0+perturb+freq gamma0+perturb gamma0+freq baseline}"
CROSS_TRAIN="${CROSS_TRAIN:-FIRE}"

CKPT_DIR="${CKPT_DIR:-workspace/checkpoints}"
GROUP="${GROUP:-fewshot_fire}"

TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-medium}"
DATA_ROOT="${DATA_ROOT:-/change/to/custom/root}"  # Set this to your Retina dataset directory.
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-4}"
SAVE_EVERY="${SAVE_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SUPPORT="${NUM_SUPPORT:-1}"
ALIGN_SCALES="${ALIGN_SCALES:-4 8 16}"
GAMMA_INIT="${GAMMA_INIT:-0.1}"
PERTURB_NOISE_STD="${PERTURB_NOISE_STD:-0.25}"
SEED="${SEED:-2026}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if ! ${PYRUN} -c "
import sys
from experiments.utils.variants import variant_spec
for name in sys.argv[1:]:
    variant_spec(name)
" ${VARIANTS}; then
  echo "Unsupported variant in VARIANTS='${VARIANTS}'; see experiments/utils/variants.py." >&2
  exit 2
fi

steps_for() {
  case "$1" in
    FIRE)   echo "${STEPS_FIRE:-1500}" ;;
    *)      echo "${NUM_STEPS:-2000}" ;;
  esac
}

completed_steps() {
  local progress="$1"
  [ -f "${progress}" ] || return 0
  sed -n 's/^[[:space:]]*"step"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${progress}" | head -1
}

run_train() {
  local variant="$1"
  local subset="$2"
  local fold="$3"
  local steps
  local ckpt_root
  local log
  local progress
  local done_steps

  steps=$(steps_for "${subset}")
  ckpt_root="${CKPT_DIR}/${GROUP}/${variant}"
  log="${ckpt_root}/${subset}/train_${variant}_${subset}_fold${fold}.log"
  mkdir -p "${ckpt_root}/${subset}"

  progress="${ckpt_root}/${subset}/fold${fold}/roma_fewshot_${subset}_progress.json"
  done_steps=$(completed_steps "${progress}")
  if [ -n "${done_steps}" ] && [ "${done_steps}" -ge "${steps}" ]; then
    echo ">>> variant=${variant} ${subset} fold${fold}: already trained ${done_steps}/${steps} steps, skipping"
    return 0
  fi

  echo "==================================================================="
  echo ">>> variant=${variant}  train=${subset}  val=${subset}  fold=${fold}  num_steps=${steps}"
  echo ">>>   ckpt -> ${ckpt_root}/${subset}/fold${fold}/roma_fewshot_${subset}_{best,latest}.pth"
  echo ">>>   log  -> ${log}"
  echo "==================================================================="
  if [ -n "${done_steps}" ]; then
    echo ">>>   resuming from step ${done_steps}"
  fi

  ${PYRUN} -u experiments/Retina/train_roma_fewshot.py \
      --dont_log_wandb \
      --train_resolution "${TRAIN_RESOLUTION}" \
      --data_root "${DATA_ROOT}" \
      --train_set "${subset}" \
      --fold "${fold}" \
      --ckpt_dir "${ckpt_root}" \
      --gpu_batch_size "${GPU_BATCH_SIZE}" \
      --num_steps "${steps}" \
      --save_every "${SAVE_EVERY}" \
      --num_workers "${NUM_WORKERS}" \
      --num_support "${NUM_SUPPORT}" \
      --variant "${variant}" \
      --align_scales ${ALIGN_SCALES} \
      --gamma_init "${GAMMA_INIT}" \
      --perturb_noise_std "${PERTURB_NOISE_STD}" \
      --seed "${SEED}" \
      ${EXTRA_ARGS} \
      2>&1 | tee -a "${log}"
  local rc=${PIPESTATUS[0]}
  if [ "${rc}" -ne 0 ]; then
    echo "!!! variant=${variant} subset=${subset} fold=${fold} FAILED (exit ${rc}), continuing"
  fi
}

for variant in ${VARIANTS}; do
  if [ "${variant}" = "zero" ]; then
    echo ">>> variant=zero is off-the-shelf RoMa; nothing to train, skipping"
    continue
  fi

  run_train "${variant}" "${CROSS_TRAIN}" none
done

echo "=== fewshot retina cross-domain train sweep done (variants=[${VARIANTS}]) ==="
