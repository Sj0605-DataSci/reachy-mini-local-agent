#!/usr/bin/env bash
# Streaming TTS front (:8003) -> rumik on vLLM (:8004). Needs torch + transformers (Mimi codec) + fastapi.
cd "$(dirname "$0")/.."
exec ${PYTHON:-python} tts_stream.py
