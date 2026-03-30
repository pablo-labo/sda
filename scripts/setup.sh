#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-specdiff_aoi}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
HF_TOKEN="${HF_TOKEN:-}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DLLM_MODEL="${DLLM_MODEL:-Efficient-Large-Model/Fast_dLLM_v2_1.5B}"
TARGET_MODEL_DIR="${TARGET_MODEL_DIR:-${REPO_ROOT}/models/Qwen2.5-7B-Instruct}"
DLLM_DIR="${DLLM_DIR:-${REPO_ROOT}/Fast_dLLM_v2_1.5B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
TIMING_OUT="${TIMING_OUT:-${OUTPUT_DIR}/system_timing.jsonl}"
DATASET_NAME="${DATASET_NAME:-gsm8k}"
NUM_QUESTIONS="${NUM_QUESTIONS:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SPEC_LEN="${SPEC_LEN:-8}"
DRAFTER_THRESHOLD="${DRAFTER_THRESHOLD:-0.9}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1,2,3}"
NUM_DRAFTERS="${NUM_DRAFTERS:-3}"
TARGET_GPU="${TARGET_GPU:-0}"
DRAFTER_GPUS="${DRAFTER_GPUS:-1 2 3}"
RUN_NAME="${RUN_NAME:-gsm8k_smoke_opt}"
SKIP_ENV_CREATE="${SKIP_ENV_CREATE:-0}"
SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

export HF_HOME
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  printf '\n[ERROR] %s\n' "$*" >&2
  exit 1
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    die "conda not found. Install Miniconda/Anaconda first, or rerun with SKIP_ENV_CREATE=1 and a ready Python environment."
  fi

  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
}

ensure_env() {
  activate_conda

  if [[ "${SKIP_ENV_CREATE}" == "1" ]]; then
    log "SKIP_ENV_CREATE=1, activating existing conda env ${ENV_NAME}"
    conda activate "${ENV_NAME}" || die "failed to activate env ${ENV_NAME}"
    return
  fi

  local sanitized_env
  sanitized_env="$(mktemp)"
  sed '/^prefix:/d' "${REPO_ROOT}/environment.yaml" > "${sanitized_env}"

  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Conda env ${ENV_NAME} already exists"
  else
    log "Creating conda env ${ENV_NAME} from environment.yaml"
    conda env create -n "${ENV_NAME}" -f "${sanitized_env}"
  fi

  rm -f "${sanitized_env}"
  conda activate "${ENV_NAME}" || die "failed to activate env ${ENV_NAME}"
}

ensure_python_packages() {
  log "Checking core Python dependencies"
  python - <<'PY'
import importlib
mods = ["torch", "transformers", "datasets", "wandb", "huggingface_hub"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
print("Python package check passed")
PY
}

ensure_gpu_visibility() {
  log "Checking CUDA visibility"
  python - <<'PY'
import torch
count = torch.cuda.device_count()
print(f"Visible CUDA devices: {count}")
if count < 4:
    raise SystemExit("Need at least 4 visible CUDA devices for the default multi-GPU smoke test")
for idx in range(count):
    print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
PY
}

ensure_hf_auth() {
  if [[ -n "${HF_TOKEN}" ]]; then
    log "Logging into Hugging Face with HF_TOKEN"
    python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face login complete")
PY
  else
    log "HF_TOKEN not set, relying on existing Hugging Face login/cache"
  fi
}

download_target_model() {
  mkdir -p "${TARGET_MODEL_DIR}"
  log "Downloading target model ${TARGET_MODEL} to ${TARGET_MODEL_DIR}"
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["TARGET_MODEL"],
    local_dir=os.environ["TARGET_MODEL_DIR"],
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("Target model download complete")
PY
}

download_dllm_repo() {
  if [[ -d "${DLLM_DIR}/.git" ]]; then
    log "Fast_dLLM repo already exists at ${DLLM_DIR}"
  else
    log "Cloning Fast_dLLM repo to ${DLLM_DIR}"
    git clone "https://huggingface.co/${DLLM_MODEL}" "${DLLM_DIR}"
  fi

  log "Switching Fast_dLLM remote to paper author's repo and pulling latest custom generation code"
  git -C "${DLLM_DIR}" remote set-url origin https://github.com/ruipeterpan/Fast_dLLM_v2_1.5B.git
  git -C "${DLLM_DIR}" pull origin
}

ensure_models() {
  if [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]]; then
    log "SKIP_MODEL_DOWNLOAD=1, using existing local model directories"
  else
    ensure_hf_auth
    download_target_model
    download_dllm_repo
  fi

  [[ -f "${TARGET_MODEL_DIR}/config.json" ]] || die "target model download seems incomplete: ${TARGET_MODEL_DIR}/config.json missing"
  [[ -f "${DLLM_DIR}/config.json" ]] || die "Fast_dLLM download seems incomplete: ${DLLM_DIR}/config.json missing"
}

run_smoke_test() {
  mkdir -p "${OUTPUT_DIR}"
  log "Starting small ${DATASET_NAME} smoke test"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  WANDB_NAME="${RUN_NAME}" \
  python "${REPO_ROOT}/failfast.py" \
    --multi_gpu \
    --num_drafters "${NUM_DRAFTERS}" \
    --target_gpu "${TARGET_GPU}" \
    --drafter_gpus ${DRAFTER_GPUS} \
    --dataset_name "${DATASET_NAME}" \
    --num_questions "${NUM_QUESTIONS}" \
    --target_model_name "${TARGET_MODEL_DIR}" \
    --dllm_dir "${DLLM_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --spec_len "${SPEC_LEN}" \
    --drafter_thresholds "${DRAFTER_THRESHOLD}" \
    --run_dllm_sf \
    --baseline_sweep \
    --overwrite \
    --timing_out "${TIMING_OUT}" \
    --optimize_spec_len
}

main() {
  log "Repo root: ${REPO_ROOT}"
  log "Environment name: ${ENV_NAME}"
  log "Target model dir: ${TARGET_MODEL_DIR}"
  log "Fast_dLLM dir: ${DLLM_DIR}"

  ensure_env
  ensure_python_packages
  ensure_gpu_visibility
  ensure_models
  run_smoke_test

  log "Smoke test finished. Outputs should be under ${OUTPUT_DIR}"
}

main "$@"
