"""Pollen Microduck adapter: same brain, different body.

Talks JSON-RPC 2.0 (newline-delimited) to `robotd` over its Unix socket (sim: `microduck/scripts/duck-sim`).
Continuous intents (robot.move / robot.head / robot.mouth) are streamed as notifications at 20 Hz, because
robotd's deadman stops the duck after 500 ms without fresh intent. Discrete intents (robot.do, robot.sound,
robot.setMode) are requests with a reply.

  DUCK_SOCK  robotd socket (default ~/.cache/duck-sim/duck-a.sock)
"""
from __future__ import annotations

import itertools, json, math, os, random, socket, threading, time
from pathlib import Path

import numpy as np

SOCK = os.environ.get("DUCK_SOCK", str(Path.home() / ".cache/duck-sim/duck-a.sock"))


class DuckClient:
    def __init__(self, path: str = SOCK, stream: bool = True):
        self.path = path
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self.s = None
        self.replies: dict[int, dict] = {}
        self.cv = threading.Condition()
        self.state: dict = {}
        self._connect()
        # continuous intents, streamed by _pump
        self.move = [0.0, 0.0, 0.0]                 # vx m/s, vy m/s, vyaw rad/s
        self.head = dict(neck_pitch=0.0, head_pitch=0.0, head_yaw=0.0, head_roll=0.0)
        self.mouth = 0.0
        if stream:  # read-only clients (the viewer) must not send intents
            threading.Thread(target=self._pump, daemon=True).start()

    def _connect(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(self.path)
        self.f = self.s.makefile("rwb")
        threading.Thread(target=self._reader, args=(self.f,), daemon=True).start()

    def _reader(self, f):
        """Drain the socket continuously: replies by id, robot.state pushes into self.state."""
        for line in f:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if "id" in m and m["id"] is not None:
                with self.cv:
                    self.replies[m["id"]] = m; self.cv.notify_all()
            elif m.get("method") == "robot.state":
                self.state = m.get("params") or {}

    def _send(self, msg: dict):
        self.f.write((json.dumps(msg) + "\n").encode()); self.f.flush()

    def call(self, method: str, params: dict | None = None, timeout: float = 5.0):
        i = next(self.ids)
        with self.lock:
            self._send({"jsonrpc": "2.0", "id": i, "method": method, **({"params": params} if params is not None else {})})
        with self.cv:
            if not self.cv.wait_for(lambda: i in self.replies, timeout):
                raise TimeoutError(f"{method}: no reply")
            m = self.replies.pop(i)
        if "error" in m:
            raise RuntimeError(m["error"])
        return m.get("result")

    def notify(self, method: str, params: dict):
        with self.lock:
            self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _pump(self):
        while True:
            try:
                vx, vy, vyaw = self.move
                self.notify("robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw})
                self.notify("robot.head", dict(self.head))
                self.notify("robot.mouth", {"open": float(self.mouth)})
            except Exception as e:
                print(f"[duck] stream error: {e}; reconnecting")
                time.sleep(1)
                try:
                    self._connect()
                except Exception:
                    pass
            time.sleep(0.05)


class DuckRobot:
    """Agent-facing tools for the duck (mirrors agent.Robot's shape)."""

    PERSONA = ("You are Microduck, a small, playful 25 cm robot duck with legs, a moving head and a beak, talking out loud "
               "with a friend. You can sit, stand, peck things off the floor with your beak, kick a ball, look around, "
               "quack, and act out feelings. Whenever asked to pick up, grab or fetch something, call peck(). Talk like a real person in casual speech: contractions, small reactions like "
               "'oh nice', 'hmm', 'haha', and sometimes a short follow-up question. Keep it to 1-2 short sentences. Reply "
               "in the user's language (English, Hindi or Hinglish). You may add <laugh> or <chuckle>. Never use markdown, "
               "lists or emojis, and never describe yourself as an AI.")

    TOOLS = [
        ("walk", "Walk in a direction for some seconds.", {"direction": ("string", ["forward", "backward", "left", "right"]),
                                                            "seconds": ("number", None)}),
        ("turn", "Turn in place.", {"direction": ("string", ["left", "right"]), "degrees": ("number", None)}),
        ("look", "Turn the head to look in a direction.", {"direction": ("string", ["left", "right", "up", "down", "center"])}),
        ("nod", "Nod yes.", {"times": ("integer", None)}),
        ("shake_head", "Shake head no.", {"times": ("integer", None)}),
        ("sit", "Sit down (or stand back up if sitting).", {}),
        ("stand", "Stand up (or sit if standing).", {}),
        ("emote", "Act out a feeling with the whole body.", {"feeling": ("string", ["excited", "laugh", "curious", "yes", "no",
                                                                                   "sad", "angry", "impatient", "mmh", "mock",
                                                                                   "defiant", "devastated"])}),
        ("kick", "Kick a ball.", {"foot": ("string", ["left", "right"])}),
        ("peck", "Pick up or grab something from the floor with your beak (you CAN do this).", {}),
        ("quack", "Make a duck sound to show a feeling.", {"feeling": ("string", ["greet", "curious", "happy", "calm", "alarm"])}),
        ("dance", "Do a little dance: wiggle and spin.", {}),
    ]

    def __init__(self, enabled: bool = True):
        self.life = None
        self.mini = None   # agent code checks `.mini` for "is a robot connected"
        self.duck = None
        if enabled:
            try:
                self.duck = DuckClient()
                self.duck.call("robot.subscribe", {})  # live state: odom height tells sitting vs standing
                self.mini = self.duck
                print(f"[duck] connected to robotd at {SOCK}")
            except Exception as e:
                print(f"[duck] not connected ({e}); is scripts/duck-sim up?")

    # ---- helpers
    def _do(self, skill: str) -> str:
        if not self.duck: return f"(no duck) {skill}"
        try:
            r = self.duck.call("robot.do", {"skill": skill}, timeout=15)
            return f"{skill}: {r}"
        except Exception as e:
            return f"{skill} refused: {e}"

    def _sound(self, tag: str):
        if self.duck:
            try: self.duck.call("robot.sound", {"tag": tag})
            except Exception as e: print(f"[duck] sound {tag}: {e}")

    def _head(self, yaw=0.0, pitch=0.0, roll=0.0, neck=0.0, dur=0.4):
        if self.duck:
            self.duck.head.update(head_yaw=math.radians(yaw), head_pitch=math.radians(pitch),
                                  head_roll=math.radians(roll), neck_pitch=math.radians(neck))
            time.sleep(dur)

    # ---- tools
    def walk(self, direction: str = "forward", seconds: float = 2.0) -> str:
        v = {"forward": (0.15, 0, 0), "backward": (-0.1, 0, 0), "left": (0, 0.08, 0), "right": (0, -0.08, 0)}.get(direction, (0, 0, 0))
        secs = max(0.5, min(6.0, float(seconds)))
        if self.duck:
            self.duck.move = list(v); time.sleep(secs); self.duck.move = [0.0, 0.0, 0.0]
        return f"walked {direction} {secs:.1f}s"

    def turn(self, direction: str = "left", degrees: float = 90) -> str:
        rate = 0.8 if direction == "left" else -0.8   # rad/s, positive = left
        secs = min(8.0, math.radians(abs(float(degrees))) / abs(rate))
        if self.duck:
            self.duck.move = [0.0, 0.0, rate]; time.sleep(secs); self.duck.move = [0.0, 0.0, 0.0]
        return f"turned {direction} {degrees} deg"

    def look(self, direction: str = "center") -> str:
        d = {"left": dict(yaw=40), "right": dict(yaw=-40), "up": dict(pitch=-25), "down": dict(pitch=25), "center": {}}
        self._head(**d.get(direction, {}), dur=0.6)
        return f"looked {direction}"

    def nod(self, times: int = 2) -> str:
        for _ in range(max(1, min(4, int(times)))):
            self._head(pitch=20, dur=0.25); self._head(pitch=-8, dur=0.25)
        self._head(dur=0.2)
        return "nodded"

    def shake_head(self, times: int = 2) -> str:
        for _ in range(max(1, min(4, int(times)))):
            self._head(yaw=25, dur=0.25); self._head(yaw=-25, dur=0.25)
        self._head(dur=0.2)
        return "shook head"

    def _sitting(self) -> bool:
        z = (self.duck.state.get("odom") or {}).get("position", [0, 0, 0.118])[2] if self.duck else 0.118
        return z < 0.09   # trunk ~0.118 m standing, ~0.06 m seated

    def sit(self) -> str:
        if self._sitting(): return "already sitting"
        r = self._do("sit_toggle"); time.sleep(3); return r

    def stand(self) -> str:
        if not self._sitting(): return "already standing"
        r = self._do("sit_toggle"); time.sleep(3); return r
    def emote(self, feeling: str = "excited") -> str: return self._do(feeling if feeling in DUCK_EMOTIONS else "excited")
    def kick(self, foot: str = "right") -> str: return self._do("kick_left" if foot == "left" else "kick_right")
    def peck(self) -> str:
        if self._sitting(): self.stand()
        return self._do("ground_pick")

    def quack(self, feeling: str = "greet") -> str:
        self._sound({"greet": "greet", "curious": "inquire", "happy": "chirp", "calm": "coo", "alarm": "alarm"}.get(feeling, "chirp"))
        return f"quacked ({feeling})"

    def dance(self, style: str = "") -> str:
        self._sound("chirp")
        for i in range(4):
            s = 1 if i % 2 else -1
            self._head(roll=18 * s, yaw=15 * s, dur=0.3)
        if self.duck:
            self.duck.move = [0.0, 0.0, 1.5]; time.sleep(2.0); self.duck.move = [0.0, 0.0, 0.0]
        self._head(dur=0.2)
        return "danced"

    # Reachy-compat no-ops / mappings used by web.py and agent.py
    def express(self, feeling: str = "happy") -> str:
        return self.life.perform(feeling, block=True) if self.life else self.quack("happy")

    def wake_up(self) -> str: return self.stand()
    def antennas(self, emotion: str = "happy") -> str: return self.quack({"sad": "calm", "curious": "curious"}.get(emotion, "happy"))
    def frame_jpeg(self): return None  # camera comes from the browser


# robotd's trained emotion skills (robot.do), discovered via robot.skills in the sim
DUCK_EMOTIONS = {"angry", "curious", "defiant", "devastated", "excited", "impatient", "laugh", "mmh", "mock", "no", "sad",
                 "yes", "yes_fast"}
# Reachy emotion names / feelings -> duck emotion skill
TO_DUCK = {"happy": "excited", "cheerful": "excited", "enthusiastic": "excited", "success": "excited", "proud": "excited",
           "excited": "excited", "laughing": "laugh", "loving": "yes", "grateful": "yes", "welcoming": "yes",
           "welcome": "yes", "yes": "yes", "understanding": "mmh", "thoughtful": "mmh", "thinking": "mmh",
           "curious": "curious", "inquiring": "curious", "attentive": "curious", "confused": "curious", "amazed": "curious",
           "surprised": "curious", "sad": "sad", "downcast": "sad", "lonely": "sad", "no": "no", "angry": "angry",
           "furious": "angry", "impatient": "impatient", "bored": "impatient", "oops": "mock", "shy": "mmh"}

# Fallback head+sound gestures (used if the emotion skill is refused): (sound, [(yaw, pitch, roll, secs), ...])
GESTURES = {
    "happy":   ("chirp",   [(0, -10, 15, .3), (0, -10, -15, .3), (0, 0, 0, .2)]),
    "excited": ("wheee",   [(20, -15, 10, .25), (-20, -15, -10, .25), (20, -15, 10, .25), (0, 0, 0, .2)]),
    "curious": ("inquire", [(15, 5, 25, .6), (0, 0, 0, .3)]),
    "sad":     ("coo",     [(0, 30, 0, .9), (0, 0, 0, .4)]),
    "welcome": ("greet",   [(0, 20, 0, .3), (0, -10, 0, .3), (0, 0, 0, .2)]),
    "surprised": ("alarm", [(0, -25, 0, .4), (0, 0, 0, .3)]),
    "thinking": ("coo",    [(30, -20, -10, .8), (0, 0, 0, .3)]),
}
ALIASES = {"cheerful": "happy", "enthusiastic": "excited", "success": "excited", "proud": "happy", "laughing": "happy",
           "loving": "happy", "grateful": "welcome", "welcoming": "welcome", "yes": "welcome", "understanding": "welcome",
           "inquiring": "curious", "attentive": "curious", "confused": "curious", "uncertain": "curious",
           "thoughtful": "thinking", "downcast": "sad", "lonely": "sad", "relief": "happy", "amazed": "surprised",
           "surprised": "surprised", "scared": "surprised", "fear": "surprised", "oops": "surprised", "shy": "curious"}


class DuckLife:
    """Same interface as life.Life (set_mode / see_face / speak_* / perform / gesture / lock / moves / mode)."""

    def __init__(self, robot: DuckRobot):
        self.robot, self.duck = robot, robot.duck
        self.lock = threading.RLock()
        self.mode = "idle"
        self.face, self.face_t = None, 0.0
        self.env: list[float] = []
        self.t0 = None
        self.streaming = False
        self.moves = {k: k for k in list(GESTURES) + list(ALIASES)}
        self.cur = np.zeros(4)  # yaw, pitch, roll, mouth
        self.glance = (0.0, 0.0, 0.0)
        if self.duck:
            threading.Thread(target=self._loop, daemon=True).start()

    def set_mode(self, m): self.mode = m
    def see_face(self, uv):
        if uv is not None: self.face, self.face_t = uv, time.time()

    def speak_start(self):
        self.env, self.t0, self.streaming, self.mode = [], None, True, "speaking"

    def speak_feed(self, pcm16k, delay: float = 0.6):
        if self.t0 is None: self.t0 = time.time() + delay
        x = np.asarray(pcm16k, dtype=np.float32)
        self.env += [float(np.sqrt(np.mean(x[k:k + 800] ** 2))) for k in range(0, len(x) - 799, 800)]

    def speak_end(self): self.streaming = False

    def perform(self, name: str, block: bool = False) -> str:
        key = name.rstrip("0123456789").replace("-", "_")
        skill = TO_DUCK.get(key)
        if skill and self.duck:  # prefer the trained whole-body emotion
            def run_skill():
                with self.lock:
                    r = self.robot._do(skill)
                    if "refused" not in r:
                        time.sleep(2.5)
            (run_skill() if block else threading.Thread(target=run_skill, daemon=True).start())
            return f"performed {skill}"
        key = key if key in GESTURES else ALIASES.get(key, "happy")
        sound, frames = GESTURES[key]

        def run():
            with self.lock:
                self.robot._sound(sound)
                for yaw, pitch, roll, secs in frames:
                    self.robot._head(yaw=yaw, pitch=pitch, roll=roll, dur=secs)
                self.cur[:3] = 0
        (run() if block else threading.Thread(target=run, daemon=True).start())
        return f"performed {key}"

    def gesture(self, fn, *a, **kw):
        with self.lock:
            out = fn(*a, **kw)
            self.cur[:3] = 0
            return out

    def _desired(self, t):
        yaw = pitch = roll = mouth = 0.0
        pitch += 3 * math.sin(2 * math.pi * t / 3.5)           # idle bob
        if self.face and t - self.face_t < 2.5:
            u, v = self.face
            yaw += float(np.clip((0.5 - u) * 80, -40, 40))
            pitch += float(np.clip((v - 0.45) * 40, -20, 20))
        else:
            gy, gp, until = self.glance
            if t > until + random.uniform(3, 7):
                self.glance = (random.uniform(-30, 30), random.uniform(-8, 8), t + random.uniform(1, 2.5))
            elif t < until:
                yaw += gy; pitch += gp
        if self.mode == "listening":
            roll += 15; pitch += 5
        elif self.mode == "thinking":
            yaw += 25; pitch -= 15
        elif self.mode == "speaking":
            t0 = self.t0 or t
            i = int((t - t0) / 0.05)
            if 0 <= i < len(self.env):
                e = min(1.0, self.env[i] / 0.08)
                mouth = e                                      # beak opens with the voice
                pitch += 8 * e - 2
                roll += 6 * math.sin((t - t0) * 1.5)
            elif i >= len(self.env) and not self.streaming:
                self.mode = "idle"
        return np.array([yaw, pitch, roll, mouth])

    def _loop(self):
        while True:
            t = time.time()
            if self.lock.acquire(blocking=False):
                try:
                    a = 0.4 if self.mode == "speaking" else 0.15
                    self.cur += a * (self._desired(t) - self.cur)
                    yaw, pitch, roll, mouth = self.cur
                    self.duck.head.update(head_yaw=math.radians(yaw), head_pitch=math.radians(pitch),
                                          head_roll=math.radians(roll), neck_pitch=0.0)
                    self.duck.mouth = float(np.clip(mouth, 0, 1))
                finally:
                    self.lock.release()
            time.sleep(max(0, 0.05 - (time.time() - t)))
