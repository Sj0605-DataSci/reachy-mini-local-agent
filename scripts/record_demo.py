"""Scripted demo recorder: sends turns to web.py, captures the sim MJPEG + streamed voice, and renders a
captioned MP4 with idle waits trimmed.

  python scripts/record_demo.py out/duck_demo.mp4 "sit down" "now stand up" ...
"""
import io, subprocess, sys, threading, time, wave
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

URL = "http://localhost:8090"
FPS, SR = 25, 24000
frames: list[tuple[float, bytes]] = []
stop = False


def grab():
    r = requests.get(URL + "/stream", stream=True, timeout=10)
    buf = b""
    for c in r.iter_content(16384):
        if stop: return
        buf += c
        while True:
            a = buf.find(b"\xff\xd8"); z = buf.find(b"\xff\xd9", a)
            if a < 0 or z < 0: break
            frames.append((time.time(), buf[a:z + 2])); buf = buf[z + 2:]


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists(): return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def wrap(d, text, f, w):
    out, line = [], ""
    for word in text.split():
        t = (line + " " + word).strip()
        if d.textlength(t, font=f) > w and line: out.append(line); line = word
        else: line = t
    return out + ([line] if line else [])


def main(out, turns):
    global stop
    threading.Thread(target=grab, daemon=True).start(); time.sleep(2)
    T0 = time.time()
    segs = []    # (start, end, user, reply) in wall time
    audio = []   # (wall time, pcm int16 bytes)
    for q in turns:
        ts = time.time()
        r = requests.post(URL + "/talk", json={"text": q}, timeout=120).json()
        first, chunks = None, []
        for c in requests.get(URL + "/speech/" + r["speech"], stream=True, timeout=120).iter_content(None):
            if first is None: first = time.time() + 0.6   # browser prebuffer
            chunks.append(c)
        pcm_b = b"".join(chunks); pcm_b = pcm_b[: len(pcm_b) // 2 * 2]
        audio.append((first or ts, pcm_b))
        pcm_len = len(pcm_b) / 2 / SR
        end = max(time.time(), (first or time.time()) + pcm_len) + 1.0
        while time.time() < end: time.sleep(0.05)
        segs.append((ts, end, q, r["reply"], first or ts))
        print(f"{q!r} -> {r['reply']!r}")
    stop = True; time.sleep(0.5)

    # timeline: keep each turn from 0.3 s before the request; squeeze the brain wait to <= 1.2 s
    W, H = Image.open(io.BytesIO(frames[-1][1])).size
    BAR = 110
    f_user, f_bot, f_title = font(22), font(22), font(18)
    tmp = Path(out).with_suffix(""); tmp.mkdir(exist_ok=True)
    k, track = 0, []
    for ts, end, q, reply, first in segs:
        keep = [(ts - 0.3, min(first, ts + 1.2)), (max(first - 0.3, ts + 1.2), end)]
        for a, b in keep:
            if b <= a: continue
            for t in np.arange(a, b, 1 / FPS):
                jpg = next((fb for ft, fb in reversed(frames) if ft <= t), frames[0][1])
                im = Image.open(io.BytesIO(jpg)).convert("RGB")
                canvas = Image.new("RGB", (W, H + BAR), (15, 17, 21)); canvas.paste(im, (0, 0))
                d = ImageDraw.Draw(canvas)
                d.text((12, 8), "Microduck · local voice agent on DGX Spark", font=f_title, fill=(200, 205, 215))
                d.text((14, H + 10), "You: " + q, font=f_user, fill=(120, 170, 255))
                if t >= first - 0.3:
                    for i, line in enumerate(wrap(d, "Duck: " + reply, f_bot, W - 28)[:2]):
                        d.text((14, H + 42 + i * 28), line, font=f_bot, fill=(255, 200, 120))
                canvas.save(tmp / f"{k:06d}.jpg", quality=90); k += 1
                track.append(t)
    # audio: place each PCM chunk on the kept timeline
    pcm = np.zeros(int(len(track) / FPS * SR) + SR, dtype=np.int16)
    starts = np.array(track)
    for t, c in audio:                                  # one joined reply per turn, placed where playback starts
        i = int(np.searchsorted(starts, t))
        if i >= len(starts): continue
        pos = int(i / FPS * SR); a = np.frombuffer(c, dtype="<i2")[: max(0, len(pcm) - int(i / FPS * SR))]
        pcm[pos:pos + len(a)] = a
    with wave.open(str(tmp / "a.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "%06d.jpg"), "-i", str(tmp / "a.wav"),
                    "-af", "loudnorm=I=-16:TP=-1.5", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out], check=True)
    print("wrote", out, f"{len(track) / FPS:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
