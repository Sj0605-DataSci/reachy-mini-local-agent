"""Browser view of the Microduck sim: mirrors robotd's live joints + odometry into an offscreen MuJoCo render (MJPEG).

Same HTTP interface as viewer.py (/stream). Run:  MUJOCO_GL=egl python duck_viewer.py 8091
  DUCK_MJCF  scene to render (default: microduck_rl scene_backlash.xml beside this repo)
"""
import io, math, os, sys, threading, time
from http.server import ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import viewer  # reuse its HTTP handler + frame buffer
from duck_robot import DuckClient

MJCF = os.environ.get("DUCK_MJCF", str(Path(__file__).parent.parent / "microduck/microduck_rl/src/mjlab_microduck/robot/microduck/scene_backlash.xml"))
ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]   # robotd joint order
W, H, FPS = viewer.W, viewer.H, viewer.FPS


def render_loop():
    while True:
        try:
            _render()
        except Exception:
            import traceback; traceback.print_exc(); time.sleep(2)


def _render():
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ORDER]
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    fadr = m.jnt_qposadr[free[0]] if free else None
    duck = DuckClient(stream=False)
    duck.call("robot.subscribe", {})
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam)
    cam.distance, cam.elevation, cam.azimuth = 0.75, -15, -30
    while True:
        t0 = time.time()
        st = duck.state
        if st.get("joints"):
            for a, q in zip(adr, st["joints"]):
                d.qpos[a] = q
        if fadr is not None and st.get("odom"):
            x, y, z = st["odom"]["position"]; yaw = st["odom"]["yaw"]
            d.qpos[fadr:fadr + 3] = [x, y, z]
            d.qpos[fadr + 3:fadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            cam.lookat[:] = [x, y, z * 0.8]
        mujoco.mj_forward(m, d)  # kinematic replay of the live robot
        viewer.apply_cam(cam)
        r.update_scene(d, cam)
        buf = io.BytesIO(); Image.fromarray(r.render()).save(buf, "JPEG", quality=80)
        with viewer.lock:
            viewer.frame = buf.getvalue()
        time.sleep(max(0, 1 / FPS - (time.time() - t0)))


if __name__ == "__main__":
    threading.Thread(target=render_loop, daemon=True).start()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    ThreadingHTTPServer(("0.0.0.0", port), viewer.H_).serve_forever()
