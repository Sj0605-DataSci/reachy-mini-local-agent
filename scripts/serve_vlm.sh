#!/usr/bin/env bash
# Vision: Qwen2.5-VL-3B (cached). Swap: VLM_REPO=LiquidAI/LFM2-VL-1.6B after `hf download`
exec "$(dirname "$0")/_vllm.sh" ${VLM_REPO:-Qwen/Qwen2.5-VL-3B-Instruct} 8011 0.25 --limit-mm-per-prompt '{"image":1}'
