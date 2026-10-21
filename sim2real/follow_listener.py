#!/usr/bin/env python3
"""follow_listener.py -- the follow toy's ears: the LAPTOP's default microphone (arecord; a Bluetooth headset set as
the system's input works) -> vosk, offline, ENGLISH and SPANISH side by side -> one JSON line per command on stdout:
{"cmd": "reach"|"grasp"|"end", "text": ..., "conf": ..., "lang": ...}

Its own PROCESS (started by toy_follow.py), so the speech decoding never competes with the 50 Hz arm loop. The G1's
own ASR is not usable from the SDK, so the robot's microphone is not used.

  en  reach | follow | release | open              es  alcanza | sigue | síguela | suelta | suéltala | abre   -> reach
      grasp | grab                                     agarra | coge | cógela | toma                         -> grasp
      finish                                           termina | fin                                         -> end

--mode both (default), per language TWO recognisers on the same audio:
  WHICH command: a grammar recogniser that knows only the command words, the fillers and [unk] -- sharp on a
      short word, but it squeezes any speech onto those words ("harvest finished" -> finish, "starting the harvest"
      -> toma);
  WAS IT A COMMAND: a free recogniser transcribing everything; in the words it heard around the command (+-SPAN_S),
      fillers aside: when it heard a command word too, nothing else may be there ("grasp", "robot, grasp it": yes;
      "let me turn to reach it", "harvest finished": no); when it misheard the command, it must be ONE word
      ("grasp" -> "crisp"); never an -ing form ("grabbing it": the robot's own line); and at least one free word
      must overlap the command (noise the free model hears as nothing: no).
  English and Spanish both hear every utterance: a command waits --agree_s for the other language and when they
  name DIFFERENT commands nothing is sent (an English "reach" must not become a Spanish "agarra").
--mode free: only the free recogniser (fewest false commands, most misses -- user 2026-10-21: "many not understood").
--mode grammar: any command word the grammar recogniser hears (the picker's way; most false commands).
A command repeated within --repeat_s is sent once. Every utterance heard is logged on stderr (listener.log).

Mic check without the robot:   python3 sim2real/follow_listener.py --test     (say the words, read what it heard)
Offline:                        python3 sim2real/follow_listener.py --wav a.wav b.wav   (16 kHz mono 16-bit)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import wave

SAMPLE_RATE = 16000
MODELS = os.path.expanduser("~/g1_fruits_picker/models/vosk")
LANGS = {
    "en": {"model": "vosk-model-small-en-us-0.15",
           #: spoken word -> command (with the small model's usual mishearings of the isolated word)
           "words": {"reach": "reach", "reached": "reach", "follow": "reach", "release": "reach", "open": "reach",
                     "grasp": "grasp", "grasped": "grasp", "grass": "grasp", "grab": "grasp",
                     "finish": "end", "finished": "end"},
           #: words that may surround a command without making it a sentence
           "filler": ["robot", "please", "now", "it", "the", "a", "an", "apple", "okay", "ok", "go", "hey", "and",
                      "then", "um", "uh", "huh", "oh", "yes", "hand", "me"],
           #: forms that are speech ABOUT the action, never the command (the robot's own line "grabbing it")
           "not": ["grabbing", "grasping", "reaching", "following", "finishing", "releasing", "opening"]},
    "es": {"model": "vosk-model-small-es-0.42",
           # the small Spanish model has no enclitic forms ("agárrala", "alcánzala"): spoken, they decode as
           # "agarra la" / "alcanza la" ("la" is a filler)
           "words": {"alcanza": "reach", "alcanzar": "reach", "sigue": "reach", "síguela": "reach",
                     "suelta": "reach", "suéltala": "reach", "abre": "reach",
                     "agarra": "grasp", "agarrar": "grasp", "coge": "grasp", "cógela": "grasp", "toma": "grasp",
                     "termina": "end", "terminar": "end", "fin": "end"},
           "filler": ["robot", "la", "el", "lo", "manzana", "por", "favor", "ahora", "eh", "ya", "vale", "mano",
                      "y", "oye"],
           "not": ["agarrando", "alcanzando", "siguiendo", "soltando", "terminando", "cogiendo", "tomando"]},
}
#: s around a grammar command in which the free words are counted
SPAN_S = 1.0
#: s of audio after a grammar command before it is judged without the free recogniser's words (it heard nothing)
FREE_WAIT_S = 1.5


def log(msg):
    sys.stderr.write(f"[listener {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


def _words(res):
    return [(str(w.get("word", "")), float(w.get("conf", 0.0)), float(w.get("start", 0.0)), float(w.get("end", 0.0)))
            for w in res.get("result", [])]


class Lang:
    """One language: its grammar and free recognisers on the same audio, and its command candidates."""

    def __init__(self, lang, mode, min_conf, max_words):
        from vosk import KaldiRecognizer, Model
        self.lang, self.mode, self.min_conf, self.max_words = lang, mode, min_conf, max_words
        L = self.L = LANGS[lang]
        model = Model(os.path.join(MODELS, L["model"]))
        self.gram = self.free = None
        if mode in ("both", "grammar"):
            vocab = sorted(L["words"]) + L["filler"] + ["[unk]"]
            self.gram = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(vocab, ensure_ascii=False))  # raw UTF-8
            self.gram.SetWords(True)
        if mode in ("both", "free"):
            self.free = KaldiRecognizer(model, SAMPLE_RATE)
            self.free.SetWords(True)
        self.free_words = []          # [(word, conf, start, end)] of the last ~10 s
        self.free_until = 0.0         # audio time the free recogniser has finished up to
        self.cands = []               # grammar commands waiting for the free words: (cmd, text, conf, start, end)

    def feed(self, data, t_audio):
        """-> [(cmd, text, conf, why)] decided in this chunk (cmd None = rejected, why says why)."""
        out = []
        if self.free is not None and self.free.AcceptWaveform(data):
            ws = [w for w in _words(json.loads(self.free.Result())) if w[0] != "[unk]"]
            if ws:
                self.free_words = [w for w in self.free_words if w[3] > t_audio - 10.0] + ws
                if self.mode == "free":
                    out.append(self._free_only(ws))
            self.free_until = t_audio
        if self.gram is not None and self.gram.AcceptWaveform(data):
            ws = _words(json.loads(self.gram.Result()))
            core = [w for w in ws if w[0] in self.L["words"]]
            heard = " ".join(f"{w[0]}({w[1]:.2f})" for w in ws)
            if core and self.mode == "grammar":
                w = core[0]
                out.append((self.L["words"][w[0]], heard, w[1], "") if w[1] >= self.min_conf else
                           (None, heard, None, f"confidence {w[1]:.2f}"))
            elif core:
                cmds = {self.L["words"][w[0]] for w in core}
                conf = min(w[1] for w in core)
                if len(cmds) != 1:
                    out.append((None, heard, None, "two different commands in one utterance"))
                elif conf < self.min_conf:
                    out.append((None, heard, None, f"confidence {conf:.2f} < {self.min_conf:g}"))
                else:
                    self.cands.append((cmds.pop(), heard, conf, core[0][2], core[-1][3]))
        # judge the waiting candidates once the free recogniser has passed them (or waited long enough)
        keep = []
        for c in self.cands:
            cmd, heard, conf, s, e = c
            if self.free_until < e + 0.05 and t_audio < e + FREE_WAIT_S:
                keep.append(c)
                continue
            near = [w for w in self.free_words if w[3] > s - SPAN_S and w[2] < e + SPAN_S]
            over = [w for w in near if w[3] > s - 0.15 and w[2] < e + 0.15]
            content = [w[0] for w in near if w[0] not in self.L["filler"]]
            hits = [x for x in content if x in self.L["words"]]
            free_txt = " ".join(w[0] for w in near) or "nothing"
            if not over:
                out.append((None, heard, None, f"the free transcription heard nothing there ('{free_txt}')"))
            elif any(x in self.L["not"] for x in content):
                out.append((None, heard, None, f"talk about the action, not the command: '{free_txt}'"))
            elif (hits and len(content) > len(hits)) or (not hits and len(content) > 1) or len(content) > self.max_words:
                out.append((None, heard, None, f"a sentence: '{free_txt}'"))
            else:
                out.append((cmd, f"{heard} / free: '{free_txt}'", conf, ""))
        self.cands = keep
        return out

    def _free_only(self, ws):
        core = [w for w in ws if w[0] not in self.L["filler"]]
        text = " ".join(w[0] for w in ws)
        cmds = {self.L["words"].get(w[0]) for w in core}
        if not core or None in cmds or len(cmds) != 1 or len(core) > 2:
            return None, text, None, "not a command"
        conf = min(w[1] for w in core)
        if conf < self.min_conf:
            return None, text, None, f"confidence {conf:.2f}"
        return cmds.pop(), text, conf, ""


class Listener:
    def __init__(self, langs, mode="both", min_conf=0.5, max_words=2, agree_s=0.6, repeat_s=1.5, test=False):
        self.langs = [Lang(lg, mode, min_conf, max_words) for lg in langs]
        self.agree_s, self.repeat_s, self.test = agree_s, repeat_s, test
        self.t_audio = 0.0
        self.pending = []             # [(audio t, lang, cmd, text, conf)] waiting for the other language
        self.vetoed_t = -1e9
        self.last = {}

    def feed(self, data):
        """16 kHz mono 16-bit audio -> [{"cmd", "text", "conf", "lang"}] to send now."""
        self.t_audio += len(data) / (2.0 * SAMPLE_RATE)
        now = self.t_audio
        for L in self.langs:
            for c, text, conf, why in L.feed(data, now):
                if c is None:
                    log(f"[{L.lang}] heard '{text}': ignored ({why})")
                    continue
                log(f"[{L.lang}] heard '{text}' -> {c}")
                other = [p for p in self.pending if p[1] != L.lang and now - p[0] <= self.agree_s]
                if any(p[2] != c for p in other):
                    log(f"  the languages disagree ({', '.join(p[1] + ':' + p[2] for p in other)} vs "
                        f"{L.lang}:{c}): nothing sent")
                    self.pending = [p for p in self.pending if p not in other]
                    self.vetoed_t = now
                    continue
                self.pending.append((now, L.lang, c, text, conf))
        sent, keep = [], []
        for p in self.pending:
            t, lg, c, text, conf = p
            confirmed = len(self.langs) == 1 or any(q is not p and q[2] == c for q in self.pending)
            if not confirmed and now - t < self.agree_s:
                keep.append(p)
                continue
            if t <= self.vetoed_t or now - self.last.get(c, -1e9) < self.repeat_s:
                continue                                  # vetoed, the other language's copy, or a repeat
            self.last[c] = now
            log(f"  -> {c.upper()} ({lg})")
            if self.test:
                sys.stderr.write(f"\n   >>> {c.upper()}\n\n")
            sent.append({"cmd": c, "text": text, "conf": round(conf, 2), "lang": lg})
        self.pending = keep
        return sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="default", help="ALSA capture device for arecord (arecord -L lists them)")
    ap.add_argument("--lang", default="en,es", help="languages listened to: en, es or en,es")
    ap.add_argument("--mode", choices=["both", "free", "grammar"], default="both")
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--max_words", type=int, default=2,
                    help="--mode both: words besides the fillers allowed around a command (the command included)")
    ap.add_argument("--agree_s", type=float, default=0.6,
                    help="s a command waits for the other language's verdict on the same utterance")
    ap.add_argument("--repeat_s", type=float, default=1.5)
    ap.add_argument("--test", action="store_true", help="also print the commands big in the terminal")
    ap.add_argument("--wav", nargs="+", default=None, help="decode these files instead of the microphone (tests)")
    a = ap.parse_args()
    try:
        from vosk import SetLogLevel
    except ImportError:
        log("vosk not installed in this Python")
        sys.exit(2)
    langs = [x.strip() for x in a.lang.split(",") if x.strip()]
    for lg in langs:
        if lg not in LANGS or not os.path.isdir(os.path.join(MODELS, LANGS[lg]["model"])):
            log(f"language '{lg}': no vosk model {os.path.join(MODELS, LANGS.get(lg, {}).get('model', '?'))}")
            sys.exit(2)
    SetLogLevel(-1)
    li = Listener(langs, a.mode, a.min_conf, a.max_words, a.agree_s, a.repeat_s, a.test)
    if a.wav:
        silence = b"\0" * 4000
        for f in a.wav:
            log(f"--- {os.path.basename(f)}")
            w = wave.open(f)
            chunks = [w.readframes(2000) for _ in range(w.getnframes() // 2000 + 1)] + [silence] * 24
            for d in chunks:
                for c in li.feed(d):
                    print(json.dumps(dict(c, file=os.path.basename(f)), ensure_ascii=False), flush=True)
        return
    cmd = ["arecord", "-q", "-D", a.device, "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
    try:
        mic = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    except FileNotFoundError:
        log("arecord not found (alsa-utils)")
        sys.exit(2)
    for lg in langs:
        log(f"[{lg}] " + ", ".join(f"{w}->{c}" for w, c in LANGS[lg]["words"].items()))
    log(f"listening on '{a.device}' ({a.mode} mode, confidence >= {a.min_conf:g})")
    print(json.dumps({"ready": True}), flush=True)
    try:
        while True:
            data = mic.stdout.read(4000)              # 125 ms of audio
            if not data:
                log("the microphone stream ended (arecord died)")
                sys.exit(3)
            for c in li.feed(data):
                print(json.dumps(c, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mic.terminate()


if __name__ == "__main__":
    main()
