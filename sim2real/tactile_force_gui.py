#!/usr/bin/env python3
# Copyright 2025 - G1 Dex3 Apple Grasp Project
"""
Window version of tactile_force_calibrate.py: calibrate the MAGNITUDE of the Dex3-1 right
hand's pads against a hanging (luggage) scale, with the hand drawn on screen. Read-only like
the terminal version: nothing is commanded, the hand only has to be powered and publishing
rt/dex3/right/state (and rt/lowstate, to watch the arm during a palm pull). Run
tactile_calibrate.py first: it names which pressure elements belong to which place.

  python3 sim2real/tactile_force_gui.py --net-if enp62s0
  python3 sim2real/tactile_force_gui.py --only palm index_dist --trials 3

The window
  * the hand (as in scripts/tactile_monitor.py) with the seven places; each shows its live
    force in N at the gain now in the file. The place to pull is ringed in orange; click a
    place to pull that one next (to redo it, or in another order).
  * a banner and lamps that follow the countdown, in step with the tones:
      three short beeps (the three dots) -> one long high tone, PULL -> two short beeps, RELEASE
  * the big button is ENTER: START zeroing, START countdown, ACCEPT reading. The Return key
    does the same. - / + change the preparation and the pull window.
  * type the scale's reading in the box (kg, or with a unit: 450g, 4.4N, 1.2lb) and ACCEPT.
  * every accepted place is saved at once (gains_counts_per_N in the same json, and the raw
    pulls next to it), so closing the window loses nothing.

How to load a phalanx, the palm, why no gains are commanded and what the scale can do: see
tactile_force_calibrate.py. The tones play on this laptop, never on the robot.
"""
import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import tactile_io as T  # noqa: E402
import tactile_force_calibrate as F  # noqa: E402
import scripts.tactile_layout as L  # noqa: E402  (the monitor's taxel positions)

P = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", outline="#d8d6cf", fill="#f4f3ef",
         nodata="#c3c2b7", lamp_off="#e6e4dc", good="#1baf7a", warn="#eda100", hot="#e8590c",
         bad="#e34948", info="#256abf", target="#ff7a00")
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PRETTY = {"thumb_prox": "THUMB, segment near the palm", "thumb_dist": "THUMB TIP",
          "middle_prox": "MIDDLE finger, segment near the palm", "middle_dist": "MIDDLE finger TIP",
          "index_prox": "INDEX finger, segment near the palm", "index_dist": "INDEX finger TIP",
          "palm": "PALM"}
W, H = 520, 470           # the hand canvas; data coordinates as in tactile_monitor (x 0..1.05, y 0.05..1.0)
MARK_R = 25


def px(x):
    return x * W / 1.05


def py(y):
    return (1.0 - y) * H / 0.95


def hexmix(c1, c2, f):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * f)):02x}" for x, y in zip(a, b))


def ramp(v):
    v = min(max(v, 0.0), 1.0) * (len(RAMP) - 1)
    i = min(int(v), len(RAMP) - 2)
    return hexmix(RAMP[i], RAMP[i + 1], v - i)


def rounded(x0, y0, x1, y1, r):
    """Points of a rounded rectangle for a smooth polygon."""
    return [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
            x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]


def place_xy():
    """{place: (x, y)} in data coordinates from the sim taxels the places stand for."""
    pos = {}
    for tx, loc in zip(L.TAXELS, T.TAXEL_LOCATION):
        pos.setdefault(loc, []).append(tx.xy)
    return {k: (float(np.mean([p[0] for p in v])), float(np.mean([p[1] for p in v]))) for k, v in pos.items()}


class App:
    def __init__(self, root, a, cal, todo, beep):
        self.root, self.a, self.cal, self.todo, self.beep = root, a, cal, todo, beep
        self.hand, self.conn_error, self.n = None, None, None
        self.phase = "connecting"
        self.state = {k: ("pending" if k in todo else "off" if cal.locations.get(k) else "unnamed")
                      for k in T.LOCATIONS}
        self.target = todo[0]
        self.k = 0                      # pull number within the place
        self.got = []                   # accepted pulls of the place being done
        self.trials = []                # every accepted pull of the session
        self.base = self.noise = self.pres = self.dead = None
        self.idx = {}
        self.disp_zero = None
        self.bs, self.zwin, self.win = [], [], []
        self.last_t = -1.0
        self.last = None                # the pull waiting for its reading
        self.sound_until, self.sound_name = 0.0, ""
        self.ticks_lit = 0
        self.last_draw = 0.0
        self.errs = set()
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(a.out)),
                                     time.strftime("tactile_force_%Y%m%d_%H%M%S.json"))
        self.pos = place_xy()
        self.prep_s = tk.DoubleVar(value=a.prep_s)
        self.pull_s = tk.DoubleVar(value=a.pull_s)
        self.unit = tk.StringVar(value={"n": "N"}.get(a.unit, a.unit))
        self.reading = tk.StringVar()
        self.sound_on = tk.BooleanVar(value=beep.enabled and not a.no_sound)
        self.build()
        for k, g in cal.gains.items():
            self.tree.set(k, "gain", f"{g:.0f}")
        threading.Thread(target=self._connect, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(10, self.loop)

    # ---------------------------------------------------------------- connection
    def _connect(self):
        try:
            self.hand = F.HandReader(self.a.net_if, self.a.topic)
        except Exception as e:  # noqa: BLE001
            self.conn_error = f"{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- layout
    def build(self):
        r = self.root
        r.title("Dex3 pad force calibration")
        r.configure(bg=P["surface"])
        r.resizable(False, False)
        left = tk.Frame(r, bg=P["surface"])
        left.grid(row=0, column=0, padx=(12, 6), pady=10, sticky="n")
        right = tk.Frame(r, bg=P["surface"])
        right.grid(row=0, column=1, padx=(6, 12), pady=10, sticky="n")

        self.status = tk.Label(left, text="", bg=P["surface"], font=("TkDefaultFont", 10, "bold"))
        self.status.pack(anchor="w")
        self.cv = tk.Canvas(left, width=W, height=H, bg=P["surface"], highlightthickness=0)
        self.cv.pack()
        self.draw_hand()
        self.joints = tk.Label(left, text="", bg=P["surface"], fg=P["ink2"], font=("TkFixedFont", 9))
        self.joints.pack(anchor="w")
        self.jwarn = tk.Label(left, text="", bg=P["surface"], fg=P["hot"], font=("TkDefaultFont", 9, "bold"))
        self.jwarn.pack(anchor="w")

        self.banner = tk.Label(right, text="", width=26, height=3, font=("TkDefaultFont", 17, "bold"),
                               wraplength=430, justify="center", fg="white", bg=P["info"])
        self.banner.pack(fill="x")
        self.sub = tk.Label(right, text="", bg=P["surface"], fg=P["ink2"], wraplength=430, justify="left",
                            font=("TkDefaultFont", 10))
        self.sub.pack(anchor="w", pady=(4, 2))

        self.lamps = tk.Canvas(right, width=430, height=96, bg=P["surface"], highlightthickness=0)
        self.lamps.pack()
        self.lamp = {}
        for i, (name, label) in enumerate((("ready", "get ready"), ("hold", "hold still"), ("pull", "PULL"),
                                           ("release", "release"))):
            cx = 50 + i * 92
            self.lamp[name] = self.lamps.create_oval(cx - 20, 8, cx + 20, 48, fill=P["lamp_off"], outline=P["outline"],
                                                     width=2)
            self.lamps.create_text(cx, 62, text=label, fill=P["ink2"], font=("TkDefaultFont", 9))
        self.dots = [self.lamps.create_oval(30 + j * 20, 76, 42 + j * 20, 88, fill=P["lamp_off"], outline=P["outline"])
                     for j in range(3)]
        self.snd = self.lamps.create_oval(396 - 20, 8, 396 + 20, 48, fill=P["lamp_off"], outline=P["outline"], width=2)
        self.lamps.create_text(396, 28, text="♪", fill=P["ink"], font=("TkDefaultFont", 16, "bold"))
        self.snd_txt = self.lamps.create_text(396, 62, text="sound", fill=P["ink2"], font=("TkDefaultFont", 9))

        self.go_btn = tk.Button(right, text="", font=("TkDefaultFont", 14, "bold"), height=2, command=self.on_primary,
                                bg=P["good"], fg="white", activebackground=P["good"], activeforeground="white")
        self.go_btn.pack(fill="x", pady=(8, 4))

        rd = tk.Frame(right, bg=P["surface"])
        rd.pack(fill="x")
        tk.Label(rd, text="Scale reading:", bg=P["surface"], font=("TkDefaultFont", 11)).pack(side="left")
        self.entry = tk.Entry(rd, textvariable=self.reading, width=9, font=("TkDefaultFont", 14))
        self.entry.pack(side="left", padx=6)
        ttk.Combobox(rd, textvariable=self.unit, values=("kg", "g", "N", "lb"), width=4, state="readonly").pack(
            side="left")
        tk.Label(rd, text="(a unit typed with the number wins)", bg=P["surface"], fg=P["ink2"],
                 font=("TkDefaultFont", 8)).pack(side="left", padx=6)
        self.review = tk.Label(right, text="", bg=P["surface"], fg=P["ink"], wraplength=430, justify="left",
                               font=("TkDefaultFont", 10))
        self.review.pack(anchor="w", pady=(4, 0))

        tm = tk.Frame(right, bg=P["surface"])
        tm.pack(fill="x", pady=(8, 0))
        self.adj = []
        for col, (label, var, step, lo) in enumerate((("Preparation", self.prep_s, 1.0, 5.0),
                                                      ("Pull window", self.pull_s, 0.5, 2.0))):
            fr = tk.Frame(tm, bg=P["surface"])
            fr.grid(row=0, column=col, padx=(0, 18))
            tk.Label(fr, text=label, bg=P["surface"], fg=P["ink2"], font=("TkDefaultFont", 9)).pack()
            row = tk.Frame(fr, bg=P["surface"])
            row.pack()
            m = tk.Button(row, text="-", width=3, command=lambda v=var, s=step, l=lo: v.set(max(l, v.get() - s)))
            m.pack(side="left")
            tk.Label(row, textvariable=var, width=5, bg=P["surface"], font=("TkDefaultFont", 13, "bold")).pack(
                side="left")
            tk.Label(row, text="s", bg=P["surface"]).pack(side="left")
            pl = tk.Button(row, text="+", width=3, command=lambda v=var, s=step: v.set(min(60.0, v.get() + s)))
            pl.pack(side="left", padx=(4, 0))
            self.adj += [m, pl]
        sf = tk.Frame(tm, bg=P["surface"])
        sf.grid(row=0, column=2)
        tk.Checkbutton(sf, text="sound", variable=self.sound_on, bg=P["surface"]).pack()
        tk.Button(sf, text="test tone", command=lambda: self.sound("go")).pack()

        ex = tk.Frame(right, bg=P["surface"])
        ex.pack(fill="x", pady=(8, 0))
        self.redo_btn = tk.Button(ex, text="Redo this pull", command=self.redo)
        self.redo_btn.pack(side="left")
        self.skip_btn = tk.Button(ex, text="Skip this place", command=self.skip)
        self.skip_btn.pack(side="left", padx=6)
        tk.Button(ex, text="Stop and save", command=self.stop).pack(side="left")

        cols = ("place", "state", "N", "counts", "gain")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=7, selectmode="browse")
        for c, w_ in zip(cols, (170, 90, 50, 70, 80)):
            self.tree.heading(c, text={"gain": "counts/N"}.get(c, c))
            self.tree.column(c, width=w_, anchor="w" if c in ("place", "state") else "e")
        self.tree.pack(fill="x", pady=(8, 0))
        for k in T.LOCATIONS:
            self.tree.insert("", "end", iid=k, values=(k, "", "", "", ""))
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.select(self.tree.selection()[0]) if self.tree.selection()
                       else None)
        self.logbox = tk.Text(right, width=56, height=7, font=("TkFixedFont", 9), bg="white", relief="solid", bd=1)
        self.logbox.pack(fill="x", pady=(6, 0))
        for tag, col in (("warn", P["hot"]), ("bad", P["bad"]), ("good", P["good"]), ("info", P["ink2"])):
            self.logbox.tag_configure(tag, foreground=col)
        self.logbox.configure(state="disabled")
        r.bind("<Return>", lambda e: self.on_primary())
        r.bind("<KP_Enter>", lambda e: self.on_primary())

    def draw_hand(self):
        c = self.cv
        kw = dict(outline=P["outline"], fill=P["fill"], width=2)
        c.create_polygon(rounded(px(0.32), py(0.56), px(0.66), py(0.28), 22), smooth=True, **kw)
        c.create_polygon(rounded(px(0.25), py(0.95), px(0.36), py(0.55), 22), smooth=True, **kw)
        c.create_polygon(rounded(px(0.48), py(0.99), px(0.59), py(0.55), 22), smooth=True, **kw)
        ang = math.radians(-35.0)
        pts = []
        for th in np.linspace(0, 2 * math.pi, 48):
            ex, ey = 0.17 * math.cos(th), 0.075 * math.sin(th)
            pts += [px(0.77 + ex * math.cos(ang) - ey * math.sin(ang)), py(0.28 + ex * math.sin(ang) + ey * math.cos(ang))]
        c.create_polygon(pts, smooth=True, **kw)
        c.create_text(px(0.49), py(0.12), text="Dex3-1 RIGHT hand (simplified, top-down)", fill=P["ink2"],
                      font=("TkDefaultFont", 9, "italic"))
        self.ring = c.create_oval(0, 0, 1, 1, outline=P["target"], width=5, state="hidden")
        self.ring_txt = c.create_text(0, 0, text="PULL HERE", fill=P["target"], font=("TkDefaultFont", 11, "bold"),
                                      state="hidden")
        self.mk, self.mk_val, self.mk_lbl = {}, {}, {}
        for loc, (x, y) in self.pos.items():
            cx, cy = px(x), py(y)
            big = MARK_R + (10 if loc == "palm" else 0)
            self.mk[loc] = c.create_oval(cx - big, cy - big, cx + big, cy + big, fill=P["nodata"], outline=P["ink"],
                                         width=2, tags=("mk",))
            self.mk_val[loc] = c.create_text(cx, cy, text="--", fill=P["ink"], font=("TkDefaultFont", 9, "bold"),
                                             tags=("mk",))
            self.mk_lbl[loc] = c.create_text(cx, cy + big + 9, text=loc.replace("_", " "), fill=P["ink2"],
                                             font=("TkDefaultFont", 8))
            for item in (self.mk[loc], self.mk_val[loc]):
                c.tag_bind(item, "<Button-1>", lambda e, l=loc: self.select(l))

    # ---------------------------------------------------------------- small helpers
    def log(self, msg, tag="info"):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", time.strftime("%H:%M:%S ") + msg + "\n", tag)
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def sound(self, name):
        if self.sound_on.get():
            self.beep.play(name)
        self.sound_name = {"tick": "beep", "go": "TONE", "end": "beep beep"}[name]
        self.sound_until = time.monotonic() + 0.35

    def set_phase(self, ph):
        self.phase = ph
        self.last_draw = 0.0           # redraw at once: the ring and the values follow the phase
        self.refresh()

    def gain(self, loc):
        return self.cal.gain_of(loc)

    def places_idx(self, loc):
        if loc in self.idx:
            return self.idx[loc]
        return [e for e in self.cal.locations.get(loc, []) if self.n is None or e < self.n]

    def hint(self):
        return F.hint_kg(self.k, self.a.trials)

    # ---------------------------------------------------------------- phases and buttons
    def refresh(self):
        ph, loc = self.phase, self.target
        busy = ph in ("baseline", "prep", "zero", "pull", "connecting")
        text, enabled, colour = {
            "connecting": ("Connecting to the hand ...", False, P["nodata"]),
            "need_zero": ("START zeroing (hands off the hand)  [Enter]", True, P["good"]),
            "baseline": ("Zeroing ...", False, P["nodata"]),
            "ready": (f"START countdown ({self.prep_s.get():g} s)  [Enter]", True, P["good"]),
            "prep": ("countdown running ...", False, P["nodata"]),
            "zero": ("holding still, zeroing ...", False, P["nodata"]),
            "pull": ("pulling: read the scale", False, P["nodata"]),
            "review": ("ACCEPT reading  [Enter]", True, P["info"]),
            "finished": ("All done: saved", False, P["nodata"]),
        }[ph]
        self.go_btn.configure(text=text, state="normal" if enabled else "disabled", bg=colour,
                              activebackground=colour)
        for b in self.adj:
            b.configure(state="disabled" if busy else "normal")
        self.redo_btn.configure(state="normal" if ph == "review" else "disabled")
        self.skip_btn.configure(state="normal" if ph in ("ready", "review", "need_zero") else "disabled")
        self.entry.configure(state="normal" if ph == "review" else "disabled")
        if ph == "need_zero":
            self.say(P["info"], "STEP 1: do not touch the hand", "Fingers open and still, strap not on yet. Then press "
                     "START: the pads are zeroed for 2.5 s.")
        elif ph == "ready":
            tot = self.a.trials
            self.say(P["info"], f"NEXT: {PRETTY[loc]}",
                     f"pull {self.k + 1} of {tot} on {loc}, aim for about {self.hint():.2f} kg "
                     f"({self.hint() * F.G:.1f} N). Put the strap on this pad (slack), scale in hand, then START. "
                     + ("The pull goes into the wrist and arm: the arm must be held or supported, not limp."
                        if loc == "palm" else ""))
        elif ph == "review":
            self.say(P["bad"], "RELEASE. Type the scale reading", "Then ACCEPT, or Redo this pull.")
        elif ph == "finished":
            self.say(P["good"], "ALL DONE", "Gains are saved. Click a place to redo it, or close the window.")
        for k in T.LOCATIONS:
            self.tree.set(k, "state", {"pending": "to do", "done": "done", "skipped": "skipped",
                                       "unnamed": "no elements", "off": "not selected",
                                       "none": "not reporting"}[self.state[k]])
        self.lights()

    def say(self, colour, big, small):
        self.banner.configure(bg=colour, text=big)
        self.sub.configure(text=small)

    def lights(self):
        ph = self.phase
        on = {"ready": ph == "prep", "hold": ph == "zero", "pull": ph == "pull", "release": ph == "review"}
        col = {"ready": P["warn"], "hold": P["hot"], "pull": P["good"], "release": P["bad"]}
        for k, item in self.lamp.items():
            self.lamps.itemconfigure(item, fill=col[k] if on[k] else P["lamp_off"])
        lit = self.ticks_lit if ph in ("prep", "zero", "pull") else 0
        for j, d in enumerate(self.dots):
            self.lamps.itemconfigure(d, fill=P["warn"] if j < lit else P["lamp_off"])

    def select(self, loc):
        if self.phase not in ("need_zero", "ready", "finished") or self.state.get(loc) not in (
                "pending", "done", "skipped"):
            return
        self.target, self.k, self.got = loc, 0, []
        if self.phase == "finished":
            self.set_phase("ready")
        else:
            self.refresh()

    def on_primary(self):
        ph = self.phase
        if ph == "need_zero":
            self.t0, self.bs = time.monotonic(), []
            self.set_phase("baseline")
            self.say(P["nodata"], "Zeroing ... do not touch the hand", "")
        elif ph == "ready":
            self.begin_prep()
        elif ph == "review":
            self.accept()

    def begin_prep(self):
        self.prep_total = float(self.prep_s.get())
        self.pull_total = float(self.pull_s.get())
        self.t_prep = time.monotonic()
        self.fired = set()
        self.ticks_lit = 0
        self.reading.set("")
        self.review.configure(text="")
        self.set_phase("prep")
        self.say(P["warn"], "GET READY", f"strap on {T.LOCATION_HELP[self.target]}, slack, scale in hand")

    def redo(self):
        if self.phase == "review":
            self.log(f"{self.target}: pull discarded, repeat it", "warn")
            self.set_phase("ready")

    def skip(self):
        if self.phase in ("ready", "review", "need_zero"):
            self.log(f"{self.target}: skipped", "warn")
            self.state[self.target] = "skipped"
            self.advance()

    def stop(self):
        self.save()
        self.log("stopped; what was accepted is saved", "good")
        self.set_phase("finished")

    def close(self):
        self.root.destroy()

    def advance(self):
        """Next place that is still to do (in the order of the list), or finished."""
        order = self.todo[self.todo.index(self.target) + 1:] + self.todo[:self.todo.index(self.target)]
        nxt = next((k for k in order if self.state[k] == "pending"), None)
        self.k, self.got = 0, []
        if nxt is None:
            self.log("every place is done", "good")
            self.set_phase("finished")
        else:
            self.target = nxt
            self.set_phase("ready")

    # ---------------------------------------------------------------- the loop
    def loop(self):
        try:
            self.step(time.monotonic())
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if msg not in self.errs:
                self.errs.add(msg)
                self.log("internal error: " + msg, "bad")
        self.root.after(10, self.loop)

    def step(self, now):
        a, h = self.a, self.hand
        if h is None:
            if self.conn_error:
                self.status.configure(text="● " + self.conn_error, fg=P["bad"])
                self.say(P["bad"], "No connection", self.conn_error)
            else:
                self.status.configure(text=f"● connecting on {a.net_if} (a fresh DDS process can take 20-35 s) ...",
                                      fg=P["warn"])
                self.say(P["nodata"], "Connecting to the hand ...", "")
            return
        f, q, t = h.read()
        if f is None:
            self.status.configure(text=f"● waiting for {a.topic} ...", fg=P["warn"])
            return
        if self.n is None:
            self.n = len(f)
            if self.cal.n_elements and self.n != self.cal.n_elements:
                self.conn_error = f"the hand reports {self.n} elements, {a.out} was made with {self.cal.n_elements}"
                self.hand = None
                return
            self.log(f"hand state: {self.n} pressure elements; {len(self.todo)} places to calibrate", "info")
            self.set_phase("need_zero")
        new = t != self.last_t
        self.last_t = t
        age = time.monotonic() - t
        if age > 1.0:
            self.status.configure(text=f"● stale: no message for {age:.1f} s", fg=P["warn"])
        else:
            self.status.configure(text="● connected, live", fg=P["good"])

        ph = self.phase
        if ph == "baseline":
            if new:
                self.bs.append(f.copy())
            if now - self.t0 >= a.base_s:
                self.finish_baseline()
        elif ph == "prep":
            remaining = self.t_prep + self.prep_total - now
            for s_ in (3, 2, 1):
                if self.prep_total >= s_ + 0.5 and remaining <= s_ and s_ not in self.fired:
                    self.fired.add(s_)
                    self.ticks_lit += 1
                    self.sound("tick")
                    self.lights()
            self.banner.configure(text=f"GET READY   {max(math.ceil(remaining), 0)}")
            if remaining <= a.zero_s:
                self.zwin = []
                self.set_phase("zero")
                self.say(P["hot"], "HOLD STILL", "strap slack, do not pull yet: the pads are being zeroed")
        elif ph == "zero":
            self.banner.configure(text=f"HOLD STILL   {max(math.ceil(self.t_prep + self.prep_total - now), 0)}")
            if new:
                self.zwin.append((t, f.copy(), q))
            if now >= self.t_prep + self.prep_total:
                self.finish_zero(now)
        elif ph == "pull":
            if new:
                self.win.append((t, f.copy(), q))
                if np.isfinite(self.aq0).all():
                    self.arm_move = max(self.arm_move, float(np.nanmax(np.abs(h.arm() - self.aq0))))
            self.banner.configure(text=f"PULL NOW!   {max(self.t_end - now, 0):.1f} s")
            if now >= self.t_end:
                self.finish_pull(now)

        if now - self.last_draw > 0.05:
            self.last_draw = now
            self.draw_live(f, q, now)

    def finish_baseline(self):
        a = self.a
        if len(self.bs) < 20:
            self.log(f"only {len(self.bs)} samples in {a.base_s} s: the hand state is not streaming", "bad")
            self.set_phase("need_zero")
            return
        self.base, self.noise = T.baseline(self.bs)
        self.pres = T.present(self.base)
        self.dead = a.dead_sigma * np.nan_to_num(self.noise)
        self.disp_zero = self.base + self.dead
        self.idx = {}
        for loc in T.LOCATIONS:
            self.idx[loc] = [e for e in self.cal.locations.get(loc, []) if e < self.n and self.pres[e]]
            if self.state[loc] == "pending" and not self.idx[loc]:
                self.state[loc] = "none"
                self.log(f"{loc}: none of its elements reports at rest, skipped", "warn")
        self.todo = [k for k in self.todo if self.state[k] != "none"]
        if not self.todo:
            self.log("nothing to calibrate", "bad")
            self.set_phase("finished")
            return
        if self.target not in self.todo:
            self.target = self.todo[0]
        self.log(f"zeroed: {int(self.pres.sum())} of {self.n} elements report at rest", "good")
        self.set_phase("ready")

    def finish_zero(self, now):
        a, loc = self.a, self.target
        if len(self.zwin) < 10:
            self.log(f"only {len(self.zwin)} samples in {a.zero_s} s: the hand state is not streaming", "bad")
            self.set_phase("ready")
            return
        self.zero, _ = T.baseline([w[1] for w in self.zwin])
        self.pre = F.counts_of(self.zero, self.base + self.dead, self.idx[loc])
        if self.pre > a.min_counts:
            self.log(f"the pads were already {self.pre:.0f} counts above the untouched zero (strap resting hard, "
                     f"or a hand on the finger): the pull is measured from there", "warn")
        self.z = self.zero + self.dead
        self.q0 = self.hand.read()[1]
        self.aq0 = self.hand.arm()
        self.arm_move = 0.0
        self.win = []
        self.t_end = now + self.pull_total
        self.disp_zero = self.z
        self.sound("go")
        self.set_phase("pull")
        self.say(P["good"], "PULL NOW!", "ramp up, hold steady, read the scale")

    def finish_pull(self, now):
        a, loc = self.a, self.target
        self.sound("end")
        self.disp_zero = self.base + self.dead
        if len(self.win) < 20:
            self.log(f"only {len(self.win)} samples in the pull window: the hand state is not streaming", "bad")
            self.set_phase("ready")
            return
        idx = self.idx[loc]
        val, dur, cv, drift, answered = F.analyse_pull(self.win, self.zero, self.z, idx, self.noise, self.q0,
                                                       a.min_counts)
        has_arm = bool(np.isfinite(self.aq0).all())
        self.last = dict(val=val, dur=dur, cv=cv, drift=drift, answered=answered,
                         arm=self.arm_move if has_arm else None)
        txt = (f"pads: {val:.0f} counts, {len(answered)} of {len(idx)} elements answered, held {dur:.1f} s "
               f"(spread {cv * 100:.0f}%), finger joints moved {drift:.3f} rad"
               + (f", arm moved {self.arm_move:.3f} rad" if loc == "palm" and has_arm else "")
               + f". At the gain now in the file ({self.gain(loc):.0f} counts/N) this reads "
                 f"{val / self.gain(loc):.2f} N.")
        self.review.configure(text=txt)
        self.log(f"{loc}: {txt}", "info")
        if val < a.min_counts:
            self.log(f"the pads barely answered (< {a.min_counts:.0f} counts): strap not on this pad, or too light. "
                     f"Redo this pull", "bad")
        if dur < 1.0:
            self.log("held less than 1 s: the scale's reading will not match this window; a steadier pull", "warn")
        if drift > a.drift_tol:
            self.log(f"the finger moved {drift:.3f} rad: the pad's contact changed during the pull", "warn")
        if loc == "palm" and has_arm and self.arm_move > a.arm_tol:
            self.log(f"the arm moved {self.arm_move:.3f} rad: it is not held stiffly enough, the strap's angle and "
                     f"the palm's contact changed during the pull", "warn")
        self.set_phase("review")
        self.entry.focus_set()

    def accept(self):
        a, loc, last = self.a, self.target, self.last
        txt = self.reading.get().strip()
        force = F.parse_force(txt, self.unit.get().lower())
        if force is None:
            self.log("type the scale's reading as a number, e.g. 0.45 (or 450g, 4.4N, 1.2lb)", "warn")
            return
        if last["val"] < a.min_counts:
            self.log("the pads did not answer in this pull: it cannot be used. Redo this pull", "bad")
            return
        if force < 0.5:
            self.log(f"{force:.2f} N is below what this scale resolves: repeat with a firmer pull", "warn")
            return
        if force > a.max_kg * F.G:
            self.log(f"{force:.1f} N is above --max_kg ({a.max_kg * F.G:.1f} N): kept, but do not go higher", "warn")
        trial = {"place": loc, "force_N": force, "counts": last["val"], "held_s": last["dur"],
                 "spread": last["cv"], "elements": len(self.idx[loc]), "answered": last["answered"],
                 "joint_drift_rad": last["drift"], "q_start": [float(x) for x in self.q0],
                 "arm_drift_rad": last["arm"], "zero_above_rest_counts": self.pre}
        self.got.append(trial)
        self.trials.append(trial)
        self.log(f"{loc}: {last['val']:.0f} counts / {force:.2f} N = {last['val'] / force:.0f} counts per N", "good")
        self.review.configure(text="")
        if len(self.got) < a.trials:
            self.k += 1
            self.set_phase("ready")
            return
        g, spread = F.fit_gain([t["counts"] for t in self.got], [t["force_N"] for t in self.got], a.fit)
        was = self.cal.gains.get(loc, self.cal.gain)
        self.cal.gains[loc] = g
        note = ("" if len(self.got) > 1 else "  (one pull: nothing to check it against)")
        if len(self.got) > 1 and spread > 0.3:
            note = "  -- WIDE: pads and scale do not agree linearly, or the pulls were not steady; repeat"
        self.log(f"{loc}: gain {g:.0f} counts per N (was {was:.0f}; {len(self.got)} pull"
                 f"{'s' if len(self.got) != 1 else ''}, spread {spread * 100:.0f}%){note}", "good" if not note.strip()
                 or "one pull" in note else "warn")
        t_last = self.got[-1]
        self.tree.set(loc, "N", f"{t_last['force_N']:.2f}")
        self.tree.set(loc, "counts", f"{t_last['counts']:.0f}")
        self.tree.set(loc, "gain", f"{g:.0f}")
        self.state[loc] = "done"
        self.save()
        self.advance()

    def save(self):
        a = self.a
        mark = "tactile_force_gui.py"
        if mark not in self.cal.notes:
            self.cal.notes = (self.cal.notes + " + " if self.cal.notes else "") + mark
        self.cal.save(a.out)
        with open(self.log_path, "w") as fh:
            json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "gains_counts_per_N": self.cal.gains,
                       "trials": self.trials, "dead_sigma": a.dead_sigma, "unit": a.unit}, fh, indent=1)

    # ---------------------------------------------------------------- drawing the live values
    def draw_live(self, f, q, now):
        a = self.a
        for loc in T.LOCATIONS:
            item, val = self.mk[loc], self.mk_val[loc]
            idx = self.places_idx(loc)
            if self.disp_zero is None or not idx or self.state[loc] in ("unnamed", "none"):
                self.cv.itemconfigure(item, fill=P["nodata"])
                self.cv.itemconfigure(val, text="--", fill=P["ink"])
                continue
            n_ = F.counts_of(f, self.disp_zero, idx) / self.gain(loc)
            nv = n_ / a.vmax
            self.cv.itemconfigure(item, fill=ramp(nv), outline=P["good"] if self.state[loc] == "done" else P["ink"])
            self.cv.itemconfigure(val, text=f"{n_:.1f}", fill="white" if nv >= 0.55 else P["ink"])
        # the place to pull: an orange ring that breathes, and a label
        loc = self.target
        if self.phase in ("need_zero", "ready", "prep", "zero", "pull", "review") and loc in self.pos:
            x, y = self.pos[loc]
            cx, cy = px(x), py(y)
            r = MARK_R + (10 if loc == "palm" else 0) + 7 + 3 * math.sin(now * 5.0)
            self.cv.coords(self.ring, cx - r, cy - r, cx + r, cy + r)
            self.cv.coords(self.ring_txt, cx, cy - r - 12)
            self.cv.itemconfigure(self.ring, state="normal")
            self.cv.itemconfigure(self.ring_txt, state="normal")
            self.cv.tag_raise(self.ring)
        else:
            self.cv.itemconfigure(self.ring, state="hidden")
            self.cv.itemconfigure(self.ring_txt, state="hidden")
        lit = now < self.sound_until
        self.lamps.itemconfigure(self.snd, fill=P["info"] if lit else P["lamp_off"])
        self.lamps.itemconfigure(self.snd_txt, text=self.sound_name if lit else "sound")
        self.joints.configure(text="joints [deg]  " + "  ".join(f"{n} {np.degrees(v):+.0f}" for n, v in zip(F.JOINTS, q)))
        away = [F.JOINTS[j] for j in range(1, 7) if abs(q[j]) > a.open_tol]
        self.jwarn.configure(text=("not at the open pose (0): " + ", ".join(away) + " -- open the fingers first")
                             if away else "")


def make(argv=None):
    """(root, app, beeper) for the command line `argv`; main() runs it."""
    ap = F.make_parser()
    ap.description = "Window version of tactile_force_calibrate.py (see its docstring for the method)."
    ap.add_argument("--vmax", type=float, default=5.0, help="N that maps to the top of the colour ramp")
    a = ap.parse_args(argv)
    F.a = a
    if a.zero_s > a.prep_s:
        sys.exit("--zero_s cannot be longer than --prep_s")
    if not os.path.exists(a.out):
        sys.exit(f"{a.out} not found: run tactile_calibrate.py first (it names the elements)")
    cal = T.TactileCalib.load(a.out)
    todo = [k for k in (a.only or F.PLACES) if cal.locations.get(k)]
    for k in (a.only or F.PLACES):
        if k not in todo:
            print(f"   {k}: no elements named in {a.out} -- redo it with tactile_calibrate.py --only {k}")
    if not todo:
        sys.exit("nothing to calibrate")
    root = tk.Tk()
    beep = F.Beeper(not a.no_sound)
    return root, App(root, a, cal, todo, beep), beep


def main(argv=None):
    root, _, beep = make(argv)
    print(f"sound: {beep.describe()}")
    root.mainloop()


if __name__ == "__main__":
    main()
