"""Browser front-end: your Mac's camera + mic -> DGX (Whisper -> LFM2.5/VLM -> rumik) -> voice in browser + live Reachy in sim.

  Server:  python web.py          (serves :8090, needs sim/llm/tts/vlm up)
  Mac:  ssh -N -L 8090:localhost:8090 <dgx>   ->  open http://localhost:8090  (localhost = mic/cam allowed)
"""
import base64, io, re, sys, threading, time, uuid, wave

import requests
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).parent))
from agent import EXPRESS, Brain, Robot, clean_reply, voice_style
from life import Life

app = FastAPI()
import os
if os.environ.get("ROBOT", "reachy") == "duck":   # Pollen Microduck (scripts/duck-sim) instead of Reachy Mini
    from duck_robot import DuckLife, DuckRobot
    robot = DuckRobot(True)
    life = DuckLife(robot)
else:
    robot = Robot(True)
    life = Life(robot.mini)
robot.life = life
if robot.mini: life.gesture(robot.wake_up)
brain = Brain(robot)

from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
YUNET = Path(__file__).parent / "out/models/yunet.onnx"
if not YUNET.exists():  # OpenCV 5 dropped Haar cascades; YuNet is the replacement
    YUNET.parent.mkdir(parents=True, exist_ok=True)
    YUNET.write_bytes(requests.get("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
                                   "face_detection_yunet_2023mar.onnx", timeout=60).content)
faces = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 240), 0.7)
turn_lock = threading.Lock()
pending: dict[str, str] = {}
TTS_STREAM = "http://localhost:8003/v1/audio/stream"
VOICE = "Ira"
last_frame = {"jpg": None, "caption_t": 0.0, "captioning": False}


def wav_pcm(b: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(b)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768, w.getframerate()


def stt(wav_b: bytes) -> str:
    a, sr = wav_pcm(wav_b)
    if len(a) < sr * 0.3 or np.abs(a).max() < 0.01:
        return ""
    return asr({"raw": a, "sampling_rate": sr}, generate_kwargs={"task": "transcribe"})["text"].strip()


def caption_async(jpg: bytes):
    def run():
        try:
            brain.scene = brain.describe(jpg)
            print(f"[eyes] {brain.scene}")
        except Exception as e:
            print(f"[eyes] caption failed: {e}")
        finally:
            last_frame["captioning"] = False
    last_frame["captioning"] = True
    last_frame["caption_t"] = time.time()
    threading.Thread(target=run, daemon=True).start()


@app.post("/frame")
async def frame(req: Request):
    """Webcam frame from the browser (~3/s): face -> gaze, and periodic scene caption."""
    jpg = await req.body()
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"face": None}
    h, w = img.shape[:2]
    faces.setInputSize((w, h))
    _, det = faces.detect(img)
    uv = None
    if det is not None and len(det):
        x, y, fw, fh = max(det[:, :4].tolist(), key=lambda d: d[2] * d[3])
        uv = ((x + fw / 2) / w, (y + fh / 2) / h)
        life.see_face(uv)
    last_frame["jpg"] = jpg
    if brain.vlm and life.mode != "speaking" and not last_frame["captioning"] and time.time() - last_frame["caption_t"] > 6 and not turn_lock.locked():
        caption_async(jpg)
    return {"face": uv}


@app.post("/state")
async def state(req: Request):
    mode = (await req.json()).get("mode", "idle")
    if not turn_lock.locked() and mode in ("idle", "listening"):
        life.set_mode(mode)
    return {"ok": True}


@app.post("/talk")
def talk(body: dict):  # sync -> runs in threadpool so /frame keeps flowing during a turn
    t0 = time.time()
    with turn_lock:  # one turn at a time
        life.set_mode("thinking")
        text = body.get("text") or stt(base64.b64decode(body["audio"]))
        if not text:
            life.set_mode("idle")
            return {"user": "", "reply": "", "route": "silence"}
        img = base64.b64decode(body["image"]) if body.get("image") else last_frame["jpg"]
        route, reply = brain.respond(text, img)
        reply = clean_reply(reply)
        emotion = "none"
        if route != "robot":  # robot route already moved via tools
            emotion = brain.pick_emotion(text, reply)
            if emotion in EXPRESS:
                life.perform(EXPRESS[emotion])  # plays while rumik synthesizes -> fills the wait
        t1 = time.time()
        sid = uuid.uuid4().hex[:8]
        pending[sid] = reply
    shown = re.sub(r"<(laugh|chuckle|sigh)>", lambda m: f"({m.group(1)})", reply)
    return {"user": text, "reply": shown, "route": route, "emotion": emotion, "speech": sid,
            "t_brain": round(t1 - t0, 2)}


@app.get("/speech/{sid}")
def speech(sid: str):
    """Stream rumik PCM to the browser; drive Reachy's speech sway with the same chunks."""
    reply = pending.pop(sid, None)
    if reply is None:
        return Response(status_code=404)
    text = re.sub(r"[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]", "", reply).strip()
    r = requests.post(TTS_STREAM, json={"input": f'<description="{voice_style(text)}"> {text}', "speaker": VOICE},
                      stream=True, timeout=120)
    sr = int(r.headers.get("X-Sample-Rate", 24000))

    def gen():
        life.speak_start()
        rest = b""
        try:
            for chunk in r.iter_content(chunk_size=None):
                data, rest = rest + chunk, b""
                if len(data) % 2:
                    data, rest = data[:-1], data[-1:]
                if not data:
                    continue
                pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
                life.speak_feed(np.interp(np.arange(0, len(pcm), sr / 16000), np.arange(len(pcm)), pcm))
                yield data
        finally:
            life.speak_end()
    return StreamingResponse(gen(), media_type="application/octet-stream", headers={"X-Sample-Rate": str(sr)})


@app.get("/stream")
async def stream():
    """Async relay of the renderer's MJPEG: a closed tab frees its connection (no threadpool leak)."""
    import aiohttp
    async def gen():
        async with aiohttp.ClientSession() as sess:
            async with sess.get("http://localhost:8091/stream") as r:
                async for chunk in r.content.iter_chunked(16384):
                    yield chunk
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/cam")
def cam(request: Request):
    """Orbit/zoom the sim view: forwarded to the renderer process."""
    try:
        requests.get("http://localhost:8091/cam?" + str(request.url.query), timeout=1)
    except requests.RequestException:
        pass
    return Response(status_code=204)


@app.get("/scene")
def scene():
    return {"scene": brain.scene, "mode": life.mode}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "web.html").read_text()


if __name__ == "__main__":
    import subprocess
    view = "duck_viewer.py" if os.environ.get("ROBOT", "reachy") == "duck" else "viewer.py"
    subprocess.Popen([sys.executable, str(Path(__file__).parent / view), "8091"])  # isolated renderer
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="warning")
