"""'Alive' layer for Reachy: one 20 Hz motion loop that owns the robot.

Modes (set by the web server as the conversation flows):
  idle      breathing + antenna drift + follows your face from the Mac webcam + random glances
  listening leans in, curious head tilt, antennas up
  thinking  looks up/away like it's thinking
  speaking  SDK speech-sway (reachy_mini.motion.speech_tapper) driven by rumik's audio, still facing you
Discrete gestures (Pollen emotions/dances libraries, tool moves) take the lock and the loop yields.
"""
from __future__ import annotations

import math, random, threading, time

import numpy as np

EMOTIONS_DS = "pollen-robotics/reachy-mini-emotions-library"
DANCES_DS = "pollen-robotics/reachy-mini-dances-library"


class Life:
    def __init__(self, mini):
        self.mini = mini
        self.lock = threading.RLock()  # whoever holds it drives the motors
        self.mode = "idle"
        self.face = None            # (u, v) in [0,1], camera image coords, + timestamp
        self.face_t = 0.0
        self.glance = (0.0, 0.0, 0.0)  # yaw, pitch, until
        self.sway: list[dict] = []
        self.env: list[float] = []
        self.sway_t0 = 0.0
        self.streaming = False
        self.cur = np.zeros(8)      # smoothed [x,y,z,roll,pitch,yaw, ant_r, ant_l]
        self.moves = {}
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves
            for ds in (EMOTIONS_DS, DANCES_DS):
                rm = RecordedMoves(ds)
                for n in rm.list_moves():
                    self.moves[n] = rm.get(n)
            print(f"[life] {len(self.moves)} recorded moves loaded")
        except Exception as e:
            print(f"[life] recorded moves unavailable: {e}")
        if mini:
            threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------ inputs
    def set_mode(self, mode: str):
        self.mode = mode

    def see_face(self, uv):
        if uv is not None:
            self.face, self.face_t = uv, time.time()

    def speak(self, pcm16k: np.ndarray):
        from reachy_mini.motion.speech_tapper import SwayRollRT
        self.sway = SwayRollRT().feed(pcm16k.astype(np.float32))
        self.sway_t0 = time.time()
        self.mode = "speaking"

    def speak_start(self):
        """Begin a streamed utterance; feed PCM chunks with speak_feed as they arrive."""
        from reachy_mini.motion.speech_tapper import SwayRollRT
        self._tapper = SwayRollRT()
        self.sway, self.env = [], []
        self.sway_t0 = None  # set on first audio chunk (+ browser prebuffer)
        self.streaming = True
        self.mode = "speaking"

    def speak_end(self):
        self.streaming = False

    def speak_feed(self, pcm16k: np.ndarray, delay: float = 0.6):
        if self.sway_t0 is None:
            self.sway_t0 = time.time() + delay  # browser starts playback after its prebuffer
        x = pcm16k.astype(np.float32)
        self.sway += self._tapper.feed(x)
        for k in range(0, len(x) - 799, 800):  # 50 ms loudness envelope
            self.env.append(float(np.sqrt(np.mean(x[k:k + 800] ** 2))))

    def perform(self, name: str, block: bool = False) -> str:
        """Play a recorded emotion/dance by name (non-blocking by default)."""
        mv = self.moves.get(name)
        if not (self.mini and mv):
            return f"no move {name}"

        def run():
            with self.lock:
                try:
                    self.mini.play_move(mv, initial_goto_duration=0.4, sound=False)
                except Exception as e:
                    print(f"[life] move {name} failed: {e}")
                self._settle()
        (run() if block else threading.Thread(target=run, daemon=True).start())
        return f"performed {name}"

    def gesture(self, fn, *a, **kw):
        """Run a hand-coded tool move (nod, look...) exclusively, then hand back to the loop."""
        with self.lock:
            out = fn(*a, **kw)
            self._settle()
            return out

    # ------------------------------------------------------------ internals
    def _settle(self):
        from reachy_mini.utils import create_head_pose
        try:
            self.mini.goto_target(head=create_head_pose(), antennas=[0, 0], duration=0.5, body_yaw=0.0)
        except Exception:
            pass
        self.cur[:] = 0

    def _desired(self, t: float) -> np.ndarray:
        x = y = z = roll = pitch = yaw = 0.0
        # breathing
        z += 2.5 * math.sin(2 * math.pi * t / 4.2)
        pitch += 1.2 * math.sin(2 * math.pi * t / 4.2 + 1)
        ar = al = 0.12 * math.sin(2 * math.pi * t / 3.1)
        # gaze: face if recent, else occasional glances
        if self.face and t - self.face_t < 2.5:
            u, v = self.face
            yaw += float(np.clip((0.5 - u) * 70, -35, 35))     # face on image-left -> turn to robot's left (+yaw)
            pitch += float(np.clip((v - 0.45) * 40, -18, 18))  # face high in frame -> look up (-pitch)
        else:
            gy, gp, until = self.glance
            if t > until + random.uniform(3, 7):
                self.glance = (random.uniform(-25, 25), random.uniform(-8, 8), t + random.uniform(1.0, 2.5))
            elif t < until:
                yaw += gy; pitch += gp
        m = self.mode
        if m == "listening":
            roll += 10; pitch += 4; z += 4; ar, al = 0.5, -0.5
        elif m == "thinking":
            yaw += 18; pitch -= 14; roll -= 6
            ar, al = 0.3 * math.sin(t * 2), -0.3 * math.sin(t * 2 + 1)
        elif m == "speaking":
            t0 = self.sway_t0 or t
            i = int((t - t0) / 0.05)
            if i < 0:
                pass  # waiting for the browser prebuffer
            elif i < len(self.env):
                e = min(1.0, self.env[i] / 0.08)                       # loudness 0..1
                e_prev = min(1.0, self.env[i - 2] / 0.08) if i >= 2 else 0
                accent = max(0.0, e - e_prev)                         # syllable onsets -> little nods
                pitch += 7 * e + 10 * accent - 3                      # head dips with the voice
                z += 5 * e - 1
                roll += 5 * math.sin((t - t0) * 1.4)                  # slow side-to-side sway
                yaw += 6 * math.sin((t - t0) * 0.7)
                if i < len(self.sway):
                    s = self.sway[i]
                    pitch += 2 * math.degrees(s["pitch_rad"]); yaw += 2 * math.degrees(s["yaw_rad"])
                    roll += 2 * math.degrees(s["roll_rad"])
                ar, al = 0.15 + 0.7 * e, -0.15 - 0.7 * e               # antennas flap with the voice
            elif not self.streaming:
                self.mode = "idle"
        return np.array([x, y, z, roll, pitch, yaw, ar, al])

    def _loop(self):
        from reachy_mini.utils import create_head_pose
        while True:
            t = time.time()
            if self.lock.acquire(blocking=False):
                try:
                    a = 0.35 if self.mode == "speaking" else 0.12
                    self.cur += a * (self._desired(t) - self.cur)
                    x, y, z, r, p, yw, ar, al = self.cur
                    self.mini.set_target(head=create_head_pose(x=x, y=y, z=z, roll=r, pitch=p, yaw=yw, mm=True, degrees=True),
                                         antennas=[float(ar), float(al)])
                except Exception:
                    pass
                finally:
                    self.lock.release()
            time.sleep(max(0, 0.05 - (time.time() - t)))
