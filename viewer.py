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
