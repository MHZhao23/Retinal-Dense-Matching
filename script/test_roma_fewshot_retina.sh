#!/bin/bash -l
# Evaluate few-shot RoMa Retina ablation arms trained on FIRE (foldnone)
# against each full target dataset, scoring every pair with --fold -1.
#
# Checkpoint layout, matching train_roma_fewshot_retina.sh:
#   ${CKPT_DIR}/${GROUP}/<variant>/<train_subset>/foldnone/roma_fewshot_<train_subset>_{best,latest}.pth
#   The `zero` arm runs without --checkpoint and uses official RoMa weights.
#
# Result layout -- one directory per arm, identical filenames inside:
#   ${OUT_DIR}/${GROUP}/${BENCH}/<variant>/<tag>.{log,json}
#
# Defaults remain overridable; use OVERWRITE=1 to replace cached results.
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

VARIANTS="${VARIANTS:-zero baseline few_shot few_shot+perturb few_shot+freq few_shot+perturb+freq gamma0+perturb+freq gamma0+perturb gamma0+freq}"

CROSS_TRAIN="${CROSS_TRAIN:-FIRE}"
CROSS_TEST="${CROSS_TEST:-OCTA MISP FA-SLO}"

CKPT_DIR="${CKPT_DIR:-workspace/checkpoints}"
GROUP="${GROUP:-fewshot_fire}"
CKPT_KIND="${CKPT_KIND:-best}"               # best | latest
CKPT_GROUP_DIR="${CKPT_DIR}/${GROUP}"

TRAIN_RESOLUTION="${TRAIN_RESOLUTION:-medium}"
SPLIT="${SPLIT:-test}"
DATA_ROOT="${DATA_ROOT:-/change/to/custom/root}"  # Set this to your Retina dataset directory.
OUT_DIR="${OUT_DIR:-workspace/eval}"
BENCH="${BENCH:-Retina}"                     # eval subdirectory, one per benchmark
NUM_SUPPORT="${NUM_SUPPORT:-10}"
SEED="${SEED:-2026}"

NUM_WORKERS="${NUM_WORKERS:-8}"
ALIGN_SCALES="${ALIGN_SCALES:-4 8 16}"
GAMMA_INIT="${GAMMA_INIT:-0.1}"
PERTURB_NOISE_STD="${PERTURB_NOISE_STD:-0.0}"
OVERWRITE="${OVERWRITE:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

BENCH_DIR="${OUT_DIR}/${GROUP}/${BENCH}"
mkdir -p "${BENCH_DIR}"

VIS="${VIS:-0}"                              # 1 to render one figure per scored pair

if ! ${PYRUN} -c "
import sys
from experiments.utils.variants import variant_spec
for name in sys.argv[1:]:
    variant_spec(name)
" ${VARIANTS}; then
  echo "Unsupported variant in VARIANTS='${VARIANTS}'; see experiments/utils/variants.py." >&2
  exit 2
fi

run_eval() {
  local variant="$1"
  local ckpt="$2"
  local fold="$3"
  local log="$4"
  local test_set="$5"
  local json="${log%.log}.json"

  local ckpt_args=""
  if [ -n "${ckpt}" ]; then
    ckpt_args="--checkpoint ${ckpt}"
  fi

  if [ -s "${json}" ] && [ "${OVERWRITE}" != "1" ]; then
    echo "--- cached: ${json} exists, skipping (OVERWRITE=1 to force re-test)"
    return 0
  fi
  local vis_args=()
  [ "${VIS}" = "1" ] && vis_args=(--vis_dir "$(dirname "${log}")")

  ${PYRUN} -u experiments/Retina/test_roma_fewshot.py \
      ${ckpt_args} \
      --test_set "${test_set}" \
      --fold "${fold}" \
      --split "${SPLIT}" \
      --data_root "${DATA_ROOT}" \
      --train_resolution "${TRAIN_RESOLUTION}" \
      --num_support "${NUM_SUPPORT}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --variant "${variant}" \
      --align_scales ${ALIGN_SCALES} \
      --gamma_init "${GAMMA_INIT}" \
      --perturb_noise_std "${PERTURB_NOISE_STD}" \
      "${vis_args[@]}" \
      --out_json "${json}" \
      ${EXTRA_ARGS} \
      2>&1 | tee "${log}"
  return "${PIPESTATUS[0]}"
}

ckpt_for() {
  local variant="$1"
  if [ "${variant}" = "zero" ]; then
    return 0
  fi
  local train_subset="$2"
  local fold_tag="$3"
  echo "${CKPT_GROUP_DIR}/${variant}/${train_subset}/fold${fold_tag}/roma_fewshot_${train_subset}_${CKPT_KIND}.pth"
}

for variant in ${VARIANTS}; do
  out_variant_dir="${BENCH_DIR}/${variant}"
  mkdir -p "${out_variant_dir}"

  ckpt=$(ckpt_for "${variant}" "${CROSS_TRAIN}" none)
  if [ -n "${ckpt}" ] && [ ! -f "${ckpt}" ]; then
    echo "--- skip cross_domain ${variant} ${CROSS_TRAIN}: missing ${ckpt}"
  else
    ckpt_label="${CKPT_KIND}"
    [ -z "${ckpt}" ] && ckpt_label="no fine-tuning"
    echo "==================================================================="
    echo ">>> [cross_domain] variant=${variant} train=${CROSS_TRAIN} (${ckpt_label}, 1-shot) -> each of [${CROSS_TEST}]"
    echo "==================================================================="
    for subset in ${CROSS_TEST}; do
      log="${out_variant_dir}/${CROSS_TRAIN}_foldnone_on_${subset}_whole.log"
      run_eval "${variant}" "${ckpt}" -1 "${log}" "${subset}" \
        || echo "!!! cross_domain ${variant} ${CROSS_TRAIN} -> ${subset} FAILED (exit $?), continuing"
    done
  fi

  echo "=== aggregating fewshot retina summary (variant=${variant}) ==="
  ${PYRUN} -u experiments/Retina/test_roma_fewshot_retina_summary.py \
      --eval_dir "${out_variant_dir}" \
      --cross_test ${CROSS_TEST} \
    || echo "!!! summary aggregation FAILED for ${variant} (exit $?)"
done

echo "=== fewshot retina cross-domain test sweep done (variants=[${VARIANTS}], logs + json in ${BENCH_DIR}) ==="

echo "=== assembling cross-arm ablation tables for ${BENCH} in ${BENCH_DIR} ==="
${PYRUN} -u experiments/summarize_ablation.py \
    --eval_dir "${OUT_DIR}/${GROUP}" \
    --benchmarks "${BENCH}" \
    --out_dir "${BENCH_DIR}" \
  || echo "!!! cross-arm ablation summary FAILED (exit $?), tables not refreshed"
