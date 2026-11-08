#!/usr/bin/env bash
# rumik-oss-1 on vLLM (OpenAI API, :8004) with the TinyAya out-of-tree plugin (./vllm-plugin).
#   RUMIK_MODEL_DIR  local model dir (default: NVFP4 export; make it with quantize_for_vllm.py from
#                    https://github.com/Sj0605-DataSci/rumik-oss-model-dissection, or point at the bf16 snapshot)
#   RUMIK_QUANT      modelopt_fp4 (default) or "" for bf16
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SNAP="$(readlink -f "${RUMIK_MODEL_DIR:?set RUMIK_MODEL_DIR to the rumik model directory}")"
ROOT="$(dirname "$(dirname "$SNAP")")"   # HF cache uses absolute symlinks -> mount the model root at the same path
QUANT=${RUMIK_QUANT-modelopt_fp4}
exec docker run --rm --gpus all --ipc=host -p 8004:8004 \
  -v "$ROOT":"$ROOT":ro -v "$HERE/vllm-plugin":/workspace/vllm-plugin \
  -e VLLM_PLUGINS=register_tinyaya -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  --entrypoint bash ${VLLM_IMAGE:-vllm/vllm-openai:v0.20.0} -c "pip install -q -e /workspace/vllm-plugin && \
  vllm serve $SNAP --served-model-name rumik --port 8004 --trust-remote-code \
  ${QUANT:+--quantization $QUANT} --dtype bfloat16 --gpu-memory-utilization 0.15 \
  --max-model-len 4096 --max-num-seqs 4"
