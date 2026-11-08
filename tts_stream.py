"""Streaming rumik TTS: vLLM (NVFP4, :8004) audio tokens -> chunked Mimi decode -> PCM stream (:8003).

Chunking follows ira-tts-webgpu PR #1: first chunk after 2 s of audio (25 Mimi frames @12.5 Hz),
then ~1 s chunks, each decoded with 16 frames of left context and the context samples trimmed.

  POST /v1/audio/stream  {"input", "speaker"}  -> raw int16 mono PCM (X-Sample-Rate header)
  POST /v1/audio/speech  same body               -> full WAV (compat with the old bundled server)
Run:  RUMIK_MODEL_DIR=/path/to/rumik python tts_stream.py   (or scripts/serve_tts.sh)
"""
import io, json, os, queue, re, threading, time, wave
from pathlib import Path

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from transformers import AutoFeatureExtractor, AutoTokenizer, MimiModel

REPO = Path(os.environ.get("RUMIK_MODEL_DIR", "rumik-oss-1-nvfp4-vllm")).expanduser()  # tokenizer + codec/
VLLM = "http://localhost:8004/v1/completions"
FIRST, STEP, CTX = 12, 12, 16  # frames (first chunk 1 s: GB10 single-stream is ~0.5x real time)
GAP = np.zeros(int(0.12 * 24000), dtype='<i2').tobytes()  # pause between phrases
AUDIO_RE = re.compile(r"<(\d+)_(\d+)>")

tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
mimi = MimiModel.from_pretrained(REPO / "codec", dtype=torch.bfloat16).eval().to("cuda")
SR = int(AutoFeatureExtractor.from_pretrained(REPO / "codec").sampling_rate)
SPF = int(SR / 12.5)  # samples per Mimi frame
END_ID = int(tok.get_vocab()["</audio>"])
app = FastAPI()


class Req(BaseModel):
    input: str
    speaker: str = "Ira"
    temperature: float = 0.8
    top_k: int = 30


def frames(req: Req):
    """Yield complete 8-codebook Mimi frames as vLLM streams audio tokens."""
    words = len(re.sub(r"<[^>]*>", " ", req.input).split())
    budget = min(2048, 250 + 90 * words)  # ~2.5 words/s at 100 tok/s, 2x margin -> no runaway babble
    body = {"model": "rumik", "prompt": f"<text>{req.speaker}: {req.input}<audio>", "max_tokens": budget,
            "temperature": req.temperature, "top_k": req.top_k, "stream": True,
            "stop_token_ids": [END_ID], "skip_special_tokens": False}
    buf, frame, expected = "", [], 0
    with requests.post(VLLM, json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            buf += json.loads(line[6:])["choices"][0]["text"]
            last = 0
            for m in AUDIO_RE.finditer(buf):
                code, q = int(m.group(1)), int(m.group(2))
                last = m.end()
                if q == expected:
                    frame.append(code); expected += 1
                    if expected == 8:
                        yield frame
                        frame, expected = [], 0
                elif q == 0:
                    frame, expected = [code], 1
                else:
                    frame, expected = [], 0
            buf = buf[last:]  # keep any partial token text


@torch.inference_mode()
def decode(fr: list[list[int]]) -> np.ndarray:
    codes = torch.tensor(fr, device="cuda").T.unsqueeze(0)  # [1, 8, T]
    return mimi.decode(codes).audio_values[0, 0].float().clamp(-1, 1).cpu().numpy()


def split_phrases(text: str) -> list[str]:
    """Sentences, long ones cut at commas; keep the first phrase short so audio starts fast."""
    m = re.match(r"^\s*(<description=[^>]*>)\s*", text)
    desc, text = (m.group(1) + " ", text[m.end():]) if m else ("", text)
    out = []
    for sent in re.split(r"(?<=[.!?।])\s+", text.strip()):
        parts, cur = re.split(r"(?<=[,;:])\s+", sent), ""
        for p in parts:
            limit = 8 if not out else 16
            if cur and len((cur + " " + p).split()) > limit:
                out.append(cur); cur = p
            else:
                cur = (cur + " " + p).strip()
        if cur: out.append(cur)
    return [desc + p for p in out if p.strip()] or [desc + text]


SLOTS = threading.Semaphore(3)  # vLLM runs max 4 seqs; earliest phrases get the bandwidth


def produce(req: Req, q: queue.Queue, prev: threading.Event | None, started: threading.Event):
    if prev: prev.wait()  # start in order
    SLOTS.acquire(); started.set()
    try:
        for f in frames(req):
            q.put(f)
    except Exception as e:
        print(f"[tts] phrase failed: {e}", flush=True)
    SLOTS.release()
    q.put(None)


def pcm_chunks(req: Req):
    t0, first, total = time.time(), True, 0
    phrases = split_phrases(req.input)
    queues = [queue.Queue() for _ in phrases]
    prev = None
    for ph, q in zip(phrases, queues):  # phrases generate concurrently (vLLM batches them), in order
        ev = threading.Event()
        threading.Thread(target=produce, args=(req.model_copy(update={"input": ph}), q, prev, ev), daemon=True).start()
        prev = ev
    for k, q in enumerate(queues):
        fr, done = [], 0
        while (f := q.get()) is not None:
            fr.append(f)
            if len(fr) - done >= (FIRST if done == 0 else STEP):
                yield from emit(fr, done)
                if first:
                    print(f"[tts] first audio {time.time() - t0:.2f}s ({len(phrases)} phrases)", flush=True); first = False
                done = len(fr)
        if len(fr) > done:
            yield from emit(fr, done)
        total += len(fr)
        if k < len(queues) - 1:
            yield GAP
    dur = total / 12.5
    print(f"[tts] {dur:.1f}s audio in {time.time() - t0:.2f}s (rtf {(time.time() - t0) / max(dur, .01):.2f})", flush=True)


def emit(fr, done):
    start = max(0, done - CTX)
    a = decode(fr[start:])[(done - start) * SPF:]
    yield (a * 32767).astype("<i2").tobytes()


@app.post("/v1/audio/stream")
def stream(req: Req):
    return StreamingResponse(pcm_chunks(req), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(SR)})


@app.post("/v1/audio/speech")
def speech(req: Req):
    pcm = b"".join(pcm_chunks(req))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
    return Response(out.getvalue(), media_type="audio/wav")


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="warning")
