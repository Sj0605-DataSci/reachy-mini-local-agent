#!/usr/bin/env bash
# Liquid LFM2.5-1.2B: router + tool-calling agent (fast, small)
exec "$(dirname "$0")/_vllm.sh" ${LLM_REPO:-LiquidAI/LFM2.5-1.2B-Instruct} 8010 0.20
