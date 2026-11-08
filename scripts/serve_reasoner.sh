#!/usr/bin/env bash
# Optional heavy reasoner like the blog: Nemotron-3-Nano-30B-A3B (NVFP4, cached)
exec "$(dirname "$0")/_vllm.sh" ${REASON_REPO:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4} 8012 0.35 --trust-remote-code
