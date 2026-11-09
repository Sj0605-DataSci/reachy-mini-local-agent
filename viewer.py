"""Browser viewer for the Reachy Mini sim over SSH (no X needed).

Mirrors the daemon's live joint state into an offscreen MuJoCo model and streams MJPEG.
  DGX:  MUJOCO_GL=egl .venv/bin/python reachy-mini-agent/viewer.py
  Mac:  ssh -N -L 8090:localhost:8090 <dgx>   then open http://localhost:8090
"""
import io, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np, requests
from PIL import Image
import reachy_mini

XML = os.path.join(os.path.dirname(reachy_mini.__file__), "descriptions/reachy_mini/mjcf/scene.xml")
STATE = "http://localhost:8000/api/state/full?with_head_joints=true"
W, H, FPS = 640, 480, 25
frame = b""
lock = threading.Lock()
CAM = {"az": None, "el": None, "dist": None, "home": None}  # live orbit camera, changed via /cam


def apply_cam(cam):
    """Seed CAM from the renderer's defaults once, then follow CAM every frame."""
    if CAM["home"] is None:
        CAM["home"] = (cam.azimuth, cam.elevation, cam.distance)
        CAM["az"], CAM["el"], CAM["dist"] = CAM["home"]
    cam.azimuth, cam.elevation, cam.distance = CAM["az"], CAM["el"], CAM["dist"]


def move_cam(q: dict):
    if CAM["home"] is None: return
    if "reset" in q:
        CAM["az"], CAM["el"], CAM["dist"] = CAM["home"]; return
    CAM["az"] = (CAM["az"] + float(q.get("daz", 0))) % 360
    CAM["el"] = max(-89.0, min(10.0, CAM["el"] + float(q.get("del", 0))))
    CAM["dist"] = max(0.2, min(3.0, CAM["dist"] * float(q.get("zoom", 1))))


def render_loop():
    while True:  # never die silently: log and rebuild the renderer
        try:
            _render()
        except Exception:
            import traceback; traceback.print_exc()
            time.sleep(2)


def _render():
    global frame
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.elevation, cam.azimuth = 0.55, -15, 160
    cam.lookat[:] = [0, 0, 0.18]
    apply_cam(cam)
    while True:
        t0 = time.time()
        try:
            s = requests.get(STATE, timeout=0.5).json()
            if s.get("head_joints"):
                d.ctrl[:7] = s["head_joints"]
            if s.get("antennas_position"):
                d.ctrl[7:9] = s["antennas_position"]
        except Exception:
            pass
        for _ in range(int(1 / FPS / m.opt.timestep)):  # let the closed-chain linkage settle on actuator targets
            mujoco.mj_step(m, d)
        apply_cam(cam)
        r.update_scene(d, cam)
        buf = io.BytesIO()
        Image.fromarray(r.render()).save(buf, "JPEG", quality=80)
        with lock:
            frame = buf.getvalue()
        time.sleep(max(0, 1 / FPS - (time.time() - t0)))


PAGE = b"""<!doctype html><title>Reachy Sim</title><body style="margin:0;background:#111;display:grid;place-items:center;height:100vh">
<img src="/stream" style="max-width:100%"></body>"""


class H_(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path.startswith("/cam"):
            from urllib.parse import parse_qs, urlparse
            move_cam({k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})
            self.send_response(204); self.end_headers(); return
        if self.path != "/stream":
            self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers()
            self.wfile.write(PAGE); return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
        try:
            while True:
                with lock:
                    f = frame
                if f:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")
                time.sleep(1 / FPS)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    threading.Thread(target=render_loop, daemon=True).start()
    print("viewer on http://localhost:8090  (Mac: ssh -N -L 8090:localhost:8090 <dgx>)")
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    ThreadingHTTPServer(("0.0.0.0", port), H_).serve_forever()
