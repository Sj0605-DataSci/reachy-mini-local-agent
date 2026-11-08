#!/usr/bin/env bash
# usage: _vllm.sh <hf_repo> <port> <gpu_mem_frac> [extra vllm args...]
set -euo pipefail
REPO=$1; PORT=$2; MEM=$3; shift 3
HF=${HF_HOME:-$HOME/.cache/huggingface}
exec docker run --rm --gpus all --ipc=host -p $PORT:$PORT \
  -v "$HF":/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  ${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0} --model "$REPO" --served-model-name "$REPO" --port $PORT \
  --gpu-memory-utilization $MEM --max-model-len 8192 --max-num-seqs 8 "$@"
