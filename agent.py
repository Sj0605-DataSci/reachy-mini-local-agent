"""Reachy Mini local agent (DGX Spark) — port of NVIDIA/HF "reachy-personal-assistant" to local models.

Pipeline:  mic/text -> STT (whisper-large-v3-turbo) -> router (LFM2.5) ->
             chit_chat  : LFM2.5 short reply
             robot      : LFM2.5 tool-calling agent -> Reachy Mini SDK (sim daemon)
             vision     : camera frame -> VLM (Qwen2.5-VL / LFM2-VL)
             reasoning  : Nemotron-3-Nano (if :8012 up) else LFM2.5
           -> TTS (rumik-oss-1, Ira) -> speaker + antenna wiggle while talking

Run:  .venv/bin/python reachy-mini-agent/agent.py            (text REPL)
      .venv/bin/python reachy-mini-agent/agent.py --mic      (Enter to talk)
"""
from __future__ import annotations

import argparse, ast, random, base64, io, json, math, re, threading, time, wave
from pathlib import Path

import numpy as np
import requests
from openai import OpenAI

LLM = ("http://localhost:8010/v1", "LiquidAI/LFM2.5-1.2B-Instruct")
VLM = ("http://localhost:8011/v1", "Qwen/Qwen2.5-VL-3B-Instruct")
REASONER = ("http://localhost:8012/v1", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4")
TTS_URL = "http://localhost:8003/v1/audio/speech"
OUT = Path(__file__).parent / "out"
PERSONA = ("You are Reachy, a small, warm, playful desktop robot with a moving head and two antennas, talking out loud "
           "with a friend. Talk like a real person in casual speech: contractions, small reactions like 'oh nice', 'hmm', "
           "'haha', and ask a short follow-up question sometimes. Keep it to 1-2 short sentences. Reply in the user's "
           "language (English, Hindi or Hinglish). You may add <laugh> or <chuckle> when something is funny, or <sigh>. "
           "Never use markdown, lists or emojis, and never describe yourself as an AI.")


def up(base: str) -> bool:
    try:
        return requests.get(base.rsplit("/v1", 1)[0] + "/health", timeout=1).ok
    except requests.RequestException:
        return False


# ---------------------------------------------------------------- robot
class Robot:
    def __init__(self, enabled: bool):
        self.mini = None
        self.life = None  # set by web.py (life.Life)
        if not enabled:
            return
        from reachy_mini import ReachyMini
        for backend in ("default", "no_media"):
            try:
                self.mini = ReachyMini(host="localhost", media_backend=backend, timeout=5)
                print(f"[robot] connected to daemon (media={backend})")
                break
            except Exception as e:  # media (gstreamer) may be unavailable on DGX
                print(f"[robot] connect media={backend} failed: {e}")

    def _pose(self, **kw):
        from reachy_mini.utils import create_head_pose
        return create_head_pose(degrees=True, mm=True, **kw)

    # ---- tools exposed to the LLM (keep schemas tight, like the blog advises)
    def look(self, direction: str = "center") -> str:
        d = {"left": dict(yaw=35), "right": dict(yaw=-35), "up": dict(pitch=-20),
             "down": dict(pitch=20), "center": {}}.get(direction, {})
        if self.mini: self.mini.goto_target(head=self._pose(**d), duration=0.8)
        return f"looked {direction}"

    def tilt_head(self, degrees: float = 15) -> str:
        deg = max(-30, min(30, float(degrees)))
        if self.mini: self.mini.goto_target(head=self._pose(roll=deg), duration=0.6)
        return f"tilted {deg} deg"

    def nod(self, times: int = 2) -> str:
        for _ in range(max(1, min(4, int(times)))):
            if self.mini:
                self.mini.goto_target(head=self._pose(pitch=15), duration=0.25)
                self.mini.goto_target(head=self._pose(pitch=-5), duration=0.25)
        if self.mini: self.mini.goto_target(head=self._pose(), duration=0.3)
        return "nodded"

    def shake_head(self, times: int = 2) -> str:
        for _ in range(max(1, min(4, int(times)))):
            if self.mini:
                self.mini.goto_target(head=self._pose(yaw=20), duration=0.25)
                self.mini.goto_target(head=self._pose(yaw=-20), duration=0.25)
        if self.mini: self.mini.goto_target(head=self._pose(), duration=0.3)
        return "shook head"

    def antennas(self, emotion: str = "happy") -> str:
        a = {"happy": [0.6, -0.6], "sad": [-1.2, 1.2], "curious": [0.3, 0.9],
             "neutral": [0, 0], "excited": [1.0, -1.0]}.get(emotion, [0, 0])
        if self.mini: self.mini.goto_target(antennas=a, duration=0.5)
        return f"antennas {emotion}"

    def express(self, feeling: str = "happy") -> str:
        if self.life and EXPRESS.get(feeling) in self.life.moves:
            return self.life.perform(EXPRESS[feeling], block=True)
        return self.antennas({"sad": "sad", "curious": "curious"}.get(feeling, "happy"))

    def dance(self, style: str = "") -> str:
        if self.life and self.life.moves:
            names = [d for d in DANCES if d in self.life.moves]
            return self.life.perform(style if style in names else random.choice(names), block=True)
        if self.mini:
            for i in range(6):
                s = 1 if i % 2 else -1
                self.mini.goto_target(head=self._pose(roll=12 * s, z=8), antennas=[0.8 * s, 0.8 * s],
                                      duration=0.35, body_yaw=0.3 * s)
            self.mini.goto_target(head=self._pose(), antennas=[0, 0], duration=0.5, body_yaw=0.0)
        return "danced"

    def sleep(self) -> str:
        if self.mini: self.mini.goto_sleep()
        return "went to sleep"

    def wake_up(self) -> str:
        if self.mini: self.mini.wake_up()
        return "woke up"

    def frame_jpeg(self) -> bytes | None:
        try:
            return self.mini.media.get_frame_jpeg() if self.mini else None
        except Exception:
            return None

    def talk_wiggle(self, seconds: float):
        """Antenna/head bob while audio plays (cheap 'speech animation')."""
        if not self.mini: return
        def run():
            t0 = time.time()
            while time.time() - t0 < seconds:
                p = math.sin((time.time() - t0) * 9)
                try:
                    self.mini.set_target(head=self._pose(pitch=3 * p), antennas=[0.35 * p, -0.35 * p])
                except Exception:
                    return
                time.sleep(0.05)
            self.mini.goto_target(head=self._pose(), antennas=[0, 0], duration=0.3)
        threading.Thread(target=run, daemon=True).start()


EXPRESS = {"happy": "cheerful1", "excited": "enthusiastic1", "laughing": "laughing1", "love": "loving1",
           "proud": "proud1", "surprised": "surprised1", "amazed": "amazed1", "curious": "curious1",
           "thinking": "thoughtful1", "confused": "confused1", "sad": "sad1", "scared": "scared1", "angry": "furious1",
           "tired": "tired1", "shy": "shy1", "grateful": "grateful1", "yes": "yes1", "no": "no1",
           "welcome": "welcoming1", "oops": "oops1", "success": "success1", "bored": "boredom1",
           "relieved": "relief1", "understanding": "understanding1", "attentive": "attentive1"}
DANCES = ["chicken_peck", "chin_lead", "dizzy_spin", "grid_snap", "groovy_sway_and_roll", "head_tilt_roll",
          "interwoven_spirals", "jackson_square", "neck_recoil", "pendulum_swing", "polyrhythm_combo",
          "sharp_side_tilt", "side_glance_flick", "side_peekaboo", "side_to_side_sway", "stumble_and_recover"]

TOOLS = [
    ("look", "Turn the head to look in a direction.", {"direction": ("string", ["left", "right", "up", "down", "center"])}),
    ("tilt_head", "Tilt head sideways, like being curious. degrees -30..30.", {"degrees": ("number", None)}),
    ("nod", "Nod yes.", {"times": ("integer", None)}),
    ("shake_head", "Shake head no.", {"times": ("integer", None)}),
    ("antennas", "Show an emotion with the antennas.", {"emotion": ("string", ["happy", "sad", "curious", "neutral", "excited"])}),
    ("dance", "Dance. Pick a style or leave empty for random.", {"style": ("string", DANCES)}),
    ("express", "Show a feeling with your whole body (expressive animation).", {"feeling": ("string", list(EXPRESS))}),
    ("sleep", "Go to sleep pose.", {}),
    ("wake_up", "Wake up from sleep.", {}),
]


def tool_schemas(tools=None):
    out = []
    for name, desc, params in (tools or TOOLS):
        props = {k: ({"type": t, "enum": e} if e else {"type": t}) for k, (t, e) in params.items()}
        out.append({"type": "function", "function": {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": []}}})
    return out


TOOL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.S)


def parse_lfm_calls(text: str):
    """LFM2 emits pythonic calls: <|tool_call_start|>[nod(times=2), antennas(emotion="happy")]<|tool_call_end|>"""
    calls = []
    for block in TOOL_RE.findall(text):
        try:
            tree = ast.parse(block.strip(), mode="eval").body
        except SyntaxError:
            continue
        for c in (tree.elts if isinstance(tree, ast.List) else [tree]):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name):
                calls.append((c.func.id, {k.arg: ast.literal_eval(k.value) for k in c.keywords}))
    return calls, TOOL_RE.sub("", text).replace("<|im_end|>", "").strip()


# ---------------------------------------------------------------- brain
def wiki_search(q: str, n: int = 3) -> list[dict]:
    """Wikipedia API fallback (reliable, keyless): search titles then fetch intro extracts."""
    H = {"User-Agent": "reachy-mini-agent/0.1 (local robot demo)"}
    api = "https://en.wikipedia.org/w/api.php"
    r = requests.get(api, params={"action": "query", "list": "search", "srsearch": q, "srlimit": n, "format": "json"},
                     headers=H, timeout=6).json()
    titles = [x["title"] for x in r.get("query", {}).get("search", [])]
    if not titles:
        return []
    r = requests.get(api, params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                                  "titles": "|".join(titles), "format": "json"}, headers=H, timeout=6).json()
    return [{"title": p["title"], "body": p.get("extract", "")[:600]} for p in r["query"]["pages"].values() if p.get("extract")]


def web_search(q: str) -> list[dict]:
    """DuckDuckGo/others via ddgs (rate-limits often) with retries, then Wikipedia."""
    hits: list[dict] = []
    try:
        from ddgs import DDGS
        for attempt in range(2):
            try:
                hits = DDGS(timeout=5).text(q, max_results=5)
                if hits: break
            except Exception as e:
                print(f"  [search] ddgs try {attempt + 1}: {str(e)[:60]}")
                time.sleep(0.3)
    except ImportError:
        pass
    if not hits:
        try:
            hits = wiki_search(q)
            print(f"  [search] wikipedia fallback -> {len(hits)}")
        except Exception as e:
            print(f"  [search] wikipedia failed: {e}")
    return hits


class Brain:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.llm = OpenAI(base_url=LLM[0], api_key="x")
        self.vlm = OpenAI(base_url=VLM[0], api_key="x") if up(VLM[0]) else None
        self.reasoner = OpenAI(base_url=REASONER[0], api_key="x") if up(REASONER[0]) else None
        self.history: list[dict] = []
        self.scene = ""  # live caption of the user's camera, refreshed by web.py
        print(f"[brain] vlm={'on' if self.vlm else 'off'} reasoner={'on' if self.reasoner else 'off (LFM fallback)'}")

    MOVE_WORDS = re.compile(r"\b(nod|shake|look|turn|tilt|dance|antenna|wiggle|move|sleep|wake|head|happy|sad|excited|curious|walk|sit|stand|kick|peck|quack|pick|grab|fetch|come here|go)\b", re.I)
    KNOW_WORDS = re.compile(r"(what do you know|do you know (about|of)|tell me about|heard (of|about)|who (is|was|are)|what is|what's|"
                            r"latest|news|today|current|price|weather|score|when (is|did|was)|where is|how much|how many|app\b|company)", re.I)
    SEE_WORDS = re.compile(r"\b(see|seeing|holding|camera|wearing|read this|in front of)\b", re.I)

    def route(self, text: str) -> str:
        if self.SEE_WORDS.search(text): return "vision"
        if self.MOVE_WORDS.search(text) and not self.KNOW_WORDS.search(text): return "robot"  # regex fast-path
        if self.KNOW_WORDS.search(text): return "search"
        r = self.llm.chat.completions.create(model=LLM[1], temperature=0, max_tokens=4, messages=[
            {"role": "system", "content": "Classify the user message into exactly one label:\n"
             "robot = asks the robot to move, look, nod, dance, show emotion, sleep or wake\n"
             "vision = asks about what the robot sees, the user's appearance, objects held up, reading text\n"
             "search = facts about real things: people, companies, apps, products, places, news, prices, events\n"
             "reasoning = math, planning, multi-step thinking\n"
             "chat = greetings, small talk, feelings\nReply with only the label."},
            {"role": "user", "content": text}])
        lab = r.choices[0].message.content.strip().lower()
        return next((l for l in ("robot", "vision", "search", "reasoning", "chat") if l in lab), "chat")

    def search(self, text: str) -> str:
        """Web search tool (DuckDuckGo via ddgs, no key) -> grounded spoken answer."""
        FILLER = r"^(hey|hi|ok|so)?\W*(reachy|richie)?\W*(do you know|what do you know|tell me|can you tell me|have you heard)?\s*(about|of)?\s*"
        PRON = r"\b(it|its|theirs|their|that|they|them|this|he|she|his|her|the|of|a|an)\b"
        strip = lambda t: re.sub(FILLER, "", t, flags=re.I).strip(" ?.!")
        q = strip(text)
        if re.search(r"\b(it|its|theirs|their|that|they|them|this|he|she|his|her)\b", text, re.I):
            prev = [m["content"] for m in self.history if m["role"] == "user"][-1:]  # resolve "theirs" etc.
            if prev: q = f"{strip(prev[0])} {q}"
        q = " ".join(re.sub(PRON, " ", q, flags=re.I).split())
        q = q or text
        hits = web_search(q)
        print(f"  [search] {q!r} -> {len(hits)} results")
        if not hits:
            return self.chat(text + "\n(Your web search found nothing. Say honestly you couldn't find it right now.)")
        notes = "\n".join(f"- {h['title']}: {h['body'][:300]}" for h in hits)
        msgs = [{"role": "system", "content": self.persona() + "\nYou just looked this up on the web. Answer from these "
                 "search results only, in your casual spoken style, 2 short sentences, no URLs:\n" + notes}]
        msgs += self.history[-4:] + [{"role": "user", "content": text}]
        r = self.llm.chat.completions.create(model=LLM[1], messages=msgs, temperature=0.3, max_tokens=90)
        return r.choices[0].message.content.strip()

    def persona(self) -> str:
        return getattr(self.robot, "PERSONA", PERSONA) + (f" Through your camera you currently see: {self.scene}. Mention it only when it's natural "
                          "(e.g. notice their smile, their mug, if they look tired)." if self.scene else "")

    def describe(self, image: bytes) -> str:
        """One-line caption of the camera frame for ambient awareness."""
        url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        r = self.vlm.chat.completions.create(model=VLM[1], max_tokens=40, temperature=0, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "In one sentence under 25 words: the person's expression, what they wear or hold, and the setting."}]}])
        return r.choices[0].message.content.strip()

    def pick_emotion(self, user: str, reply: str) -> str:
        """Choose a body emotion to perform alongside the spoken reply (guided decoding over the library)."""
        opts = [k for k in EXPRESS if k not in ("no", "tired", "bored", "scared", "angry")]
        r = self.llm.chat.completions.create(model=LLM[1], temperature=0, max_tokens=6, messages=[
            {"role": "system", "content": "You pick the body language of a small expressive robot. Choose the emotion that best "
             "fits how the robot feels while saying its reply. Be expressive: excited about fun topics, curious about questions, "
             "understanding when the user shares something, laughing at jokes."},
            {"role": "user", "content": f"User said: {user}\nRobot replies: {reply}\nEmotion:"}],
            extra_body={"structured_outputs": {"choice": opts}})
        return (r.choices[0].message.content or "none").strip()

    def chat(self, text, client=None, model=LLM[1], max_tokens=120):
        msgs = [{"role": "system", "content": self.persona()}] + self.history[-8:] + [{"role": "user", "content": text}]
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if client is self.reasoner else {}
        r = (client or self.llm).chat.completions.create(model=model, messages=msgs, temperature=0.4,
                                                         max_tokens=max_tokens, extra_body=extra)
        return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()

    def act(self, text: str) -> str:
        msgs = [{"role": "system", "content": self.persona() + " Use tools to move. You may call several tools."},
                {"role": "user", "content": text}]
        done = []
        for _ in range(2):  # one tool round, then react (small models repeat calls if looped)
            r = self.llm.chat.completions.create(
                model=LLM[1], messages=msgs, tools=tool_schemas(getattr(self.robot, "TOOLS", TOOLS)), tool_choice="none", temperature=0.1,
                max_tokens=200, extra_body={"skip_special_tokens": False})
            raw = r.choices[0].message.content or ""
            calls, said = parse_lfm_calls(raw)
            if not calls:
                if done:  # speak naturally about it instead of a robotic tool summary
                    return self.chat(f"{text}\n(You just did that with your body: {', '.join(done)}. "
                                     "React naturally in one short sentence, don't list what you did.)", max_tokens=50)
                return said or "Hmm, I'm not sure how to do that."
            msgs.append({"role": "assistant", "content": raw})
            uniq = []
            for c in calls:
                if c not in uniq: uniq.append(c)
            for name, args in uniq:
                fn = getattr(self.robot, name, None)
                if not (fn and name in {t[0] for t in getattr(self.robot, "TOOLS", TOOLS)}):
                    res = f"unknown tool {name}"
                elif self.robot.life and name not in ("express", "dance"):
                    res = self.robot.life.gesture(fn, **args)
                else:
                    res = fn(**args)
                print(f"  [tool] {name}({args}) -> {res}")
                done.append(name)
                msgs.append({"role": "tool", "content": json.dumps({"result": res})})
            return self.chat(f"{text}\n(You just did that with your body: {', '.join(dict.fromkeys(done))}. "
                             "React naturally in one short sentence, don't list what you did.)", max_tokens=50)
        return "All done!"

    def see(self, text: str, image: bytes | None) -> str:
        if not self.vlm:
            return "My eyes aren't switched on right now."
        if image is None:
            return "I can't get a camera frame right now."
        url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        r = self.vlm.chat.completions.create(model=VLM[1], max_tokens=120, temperature=0.2, messages=[
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": text}]}])
        return r.choices[0].message.content.strip()

    def respond(self, text: str, image: bytes | None = None) -> tuple[str, str]:
        route = self.route(text)
        if route == "robot":
            reply = self.act(text)
        elif route == "vision":
            reply = self.see(text, image or self.robot.frame_jpeg())
        elif route == "search":
            reply = self.search(text)
        elif route == "reasoning" and self.reasoner:
            reply = self.chat(text, self.reasoner, REASONER[1], 300)
        else:
            reply = self.chat(text)
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return route, reply


# ---------------------------------------------------------------- ears / mouth
class Ears:
    def __init__(self):
        import torch
        from transformers import pipeline
        self.asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
                            torch_dtype=torch.float16, device="cuda")

    def listen(self, seconds: float = 5.0, sr: int = 16000) -> str:
        import sounddevice as sd
        input("  [mic] press Enter and speak...")
        a = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32"); sd.wait()
        return self.asr({"raw": a[:, 0], "sampling_rate": sr})["text"].strip()


def clean_reply(text: str, max_sentences: int = 2) -> str:
    """Spoken-style cleanup: rumik tags only (<laugh>/<chuckle>/<sigh>), no emojis/markdown, short."""
    t = re.sub(r"[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F*#_`]", "", text)
    t = re.sub(r"[(\[<]\s*(laugh|laughs|laughing|chuckle|chuckles|sigh|sighs)\s*[)\]>]",
               lambda m: "<" + {"laughs": "laugh", "laughing": "laugh", "chuckles": "chuckle", "sighs": "sigh"}.get(m.group(1).lower(), m.group(1).lower()) + ">", t, flags=re.I)
    t = re.sub(r"</[^>]*>", "", t)
    t = re.sub(r"<(?!(laugh|chuckle|sigh)>)[^>]*>", "", t)
    t = " ".join(t.split())
    return " ".join(re.split(r"(?<=[.!?\u0964])\s+", t)[:max_sentences]).strip()


def voice_style(text: str) -> str:
    t = text.lower()
    if "<laugh>" in t or "<chuckle>" in t or "haha" in t: mood = "happy, amused"
    elif "<sigh>" in t or "sorry" in t: mood = "soft, gentle"
    elif "?" in t: mood = "curious, warm"
    else: mood = "warm, friendly, cheerful"
    return f"{mood}, Indian accent, natural conversational pace"


def tts_bytes(text: str) -> bytes | None:
    text = re.sub(r"[\U0001F000-\U0001FFFF☀-➿️]", "", text).strip()  # emojis break TTS
    try:
        r = requests.post(TTS_URL, json={"input": f'<description="{voice_style(text)}"> {text}', "speaker": "Ira"},
                          timeout=120)
        r.raise_for_status()
        return r.content
    except requests.RequestException as e:
        print(f"  [tts] off ({e.__class__.__name__})"); return None


def wav_seconds(b: bytes) -> float:
    with wave.open(io.BytesIO(b)) as w:
        return w.getnframes() / w.getframerate()


def speak(text: str, robot: Robot, n: int) -> Path | None:
    content = tts_bytes(text)
    if content is None: return None
    path = OUT / f"reply_{n:03d}.wav"; path.write_bytes(content)
    with wave.open(io.BytesIO(content)) as w:
        sr, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    robot.talk_wiggle(len(pcm) / sr)
    try:
        import sounddevice as sd
        sd.play(pcm.astype(np.float32) / 32768, sr); sd.wait()
    except Exception:
        time.sleep(len(pcm) / sr)  # no speaker on the DGX: wav is saved, still animate
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="speak instead of typing (needs input device)")
    ap.add_argument("--no-robot", action="store_true", help="run brain without the sim daemon")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--image", help="jpeg to use for vision questions instead of the robot camera")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    assert up(LLM[0]), "LLM not up: run scripts/serve_llm.sh"
    robot = Robot(not a.no_robot)
    if robot.mini: robot.wake_up()
    brain, ears = Brain(robot), (Ears() if a.mic else None)
    img = Path(a.image).read_bytes() if a.image else None
    n = 0
    while True:
        try:
            text = ears.listen() if ears else input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text: continue
        if ears: print(f"you> {text}")
        t0 = time.time()
        route, reply = brain.respond(text, img)
        print(f"reachy [{route} {time.time()-t0:.1f}s]> {reply}")
        if not a.no_tts:
            n += 1; p = speak(reply, robot, n)
            if p: print(f"  [tts] {p}")


if __name__ == "__main__":
    main()
