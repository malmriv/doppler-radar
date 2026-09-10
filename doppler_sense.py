#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acoustic Doppler sonar, proof of concept.

The speaker emits a continuous, near-inaudible tone (~20 kHz). The microphone
picks it up over the direct path -- the carrier: enormous and perfectly steady
-- plus every echo in the room. Anything standing still reflects at the same
frequency; anything moving sends the echo back shifted:

    Δf = 2 · v · f0 / c        (c ≈ 343 m/s)

An object moving at 30 cm/s produces Δf ≈ 35 Hz on a 20 kHz carrier. In
relative terms that is a rounding error (0.17 %), but next to a spectral line
as clean as the carrier it is six FFT bins and plainly visible.

So the detector is: FFT with a very-low-sidelobe window (Blackman-Harris),
measure the energy in the sidebands around the carrier, normalise it by the
carrier power, and compare against the baseline of the empty room.

That normalisation is what lets the whole thing survive the automatic gain
control of a laptop's built-in microphone: when the system raises or lowers the
input gain, carrier and sidebands move together and the ratio never notices.

Honest limit of the physics: Doppler detects MOTION, not presence. A perfectly
still object produces no sidebands. It does perturb the carrier amplitude a
little, which is shown as a secondary metric, but that is far less reliable.

Usage:
    python doppler_sense.py                 # calibrate and detect
    python doppler_sense.py --auto-freq     # find the best carrier first
    python doppler_sense.py --list-devices
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

C_SOUND = 343.0  # m/s


# ---------------------------------------------------------------- utilities

def blackman_harris(n: int) -> np.ndarray:
    """Four-term window: sidelobes down at -92 dB.

    Essential here. With a plain Hann (-31 dB) the carrier's spectral leakage
    would bury the very sidebands we are trying to measure.
    """
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    w = 2 * np.pi * np.arange(n) / (n - 1)
    return a[0] - a[1] * np.cos(w) + a[2] * np.cos(2 * w) - a[3] * np.cos(3 * w)


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-30))


BLOCKS = " ▁▂▃▄▅▆▇█"
GREEN, CYAN, GREY, BOLD, OFF = "\033[92m", "\033[96m", "\033[90m", "\033[1m", "\033[0m"
YELLOW, DIM = "\033[93m", "\033[2m"
# Braille spinner, tick and shade all have unambiguous width, so no terminal
# renders them double and pushes these lines into a wrap that would break the
# carriage returns the animation relies on.
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def disable_colour():
    """Strip the escapes when nobody is there to render them.

    Redirect the output to a file or a pipe and the colour codes are not
    colour any more, just noise in the middle of the text.
    """
    global GREEN, CYAN, GREY, BOLD, OFF, YELLOW, DIM
    GREEN = CYAN = GREY = BOLD = OFF = YELLOW = DIM = ""


def vmeter(v: float, vmax: float, half: int = 16) -> str:
    """Bidirectional meter: zero at the centre, approaching to the right."""
    frac = min(max(v / vmax, -1.0), 1.0)
    n = int(round(abs(frac) * half))
    if v >= 0:
        left, right = "·" * half, GREEN + "█" * n + OFF + "·" * (half - n)
    else:
        left, right = "·" * (half - n) + CYAN + "█" * n + OFF, "·" * half
    return left + BOLD + "│" + OFF + right


# Braille dot bits: the left column, top to bottom, is dots 1-2-3-7, and the
# right column 4-5-6-8. Two dot columns per character is what buys the extra
# horizontal resolution, since every cell then holds two samples.
BRAILLE = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))
BRAILLE_BLANK = 0x2800


def history_braille(hist, vmax, width, rows):
    """History as braille bars growing away from a zero axis.

    Four dot rows per character and two samples per column: sixteen levels a
    side at four rows, and twice the history of a block renderer in the same
    width. Sixteen levels over a full scale of 1 m/s is 0.06 m/s a level,
    which lands about on the noise floor of the smoothed estimate, so this is
    as fine as the reading deserves to be drawn.

    Braille is also the only graphic block used here with an unambiguous
    width, so no terminal renders it double and clips the row.
    """
    n = max(min(width, len(hist) // 2), 8)
    vals = list(hist)[-n * 2:]
    vals = [0.0] * (n * 2 - len(vals)) + vals

    up = [[BRAILLE_BLANK] * n for _ in range(rows)]
    dn = [[BRAILLE_BLANK] * n for _ in range(rows)]
    for i, v in enumerate(vals):
        cell, col = divmod(i, 2)
        dots = min(abs(v) / vmax, 1.0) * rows * 4
        rowset = up if v < 0 else dn
        for r in range(rows):
            f = int(round(min(max(dots - 4 * r, 0.0), 4.0)))
            if f == 0:
                break
            for k in range(f):
                # Above the axis the bar grows from the bottom dot upward,
                # below it from the top dot down, so both leave the line.
                rowset[r][cell] |= BRAILLE[col][3 - k if v < 0 else k]

    def render(cells, colour):
        return colour + "".join(chr(c) for c in cells) + OFF

    return ([render(up[r], CYAN) for r in range(rows - 1, -1, -1)]
            + ["-" * n]
            + [render(dn[r], GREEN) for r in range(rows)])


def history_blocks(hist, vmax, width, rows):
    """The same graph in half-cell blocks: coarser, but bolder to read."""
    n = max(min(width, len(hist)), 8)
    vals = list(hist)[-n:]
    vals = [0.0] * (n - len(vals)) + vals

    up = [[" "] * n for _ in range(rows)]
    dn = [[" "] * n for _ in range(rows)]
    for i, v in enumerate(vals):
        half = min(abs(v) / vmax, 1.0) * rows * 2.0
        rowset, part, col = (up, "▄", CYAN) if v < 0 else (dn, "▀", GREEN)
        for r in range(rows):
            if half >= 2 * (r + 1):
                rowset[r][i] = col + "█" + OFF
            elif half >= 2 * r + 1:
                rowset[r][i] = col + part + OFF
            else:
                break

    return (["".join(up[r]) for r in range(rows - 1, -1, -1)]
            + ["-" * n]
            + ["".join(dn[r]) for r in range(rows)])


def history_graph(hist, vmax, width, rows, blocks=False):
    """Wrap a renderer's rows in the left gutter that labels the axis."""
    body = (history_blocks if blocks else history_braille)(
        hist, vmax, width, rows)
    out = []
    for i, row in enumerate(body):
        if i == 0:
            gutter = "    " + CYAN + "away" + OFF + "   "
        elif i == rows:
            gutter = "  history  "
        elif i == len(body) - 1:
            gutter = "  " + GREEN + "toward" + OFF + "   "
        else:
            gutter = " " * 11
        out.append(gutter + row)
    return out


def term_size():
    ts = shutil.get_terminal_size((80, 24))
    return ts.columns, ts.lines


def build_panel(width, height, smooth, base, vshow, vmax, hist, state,
                carrier_d, f0, blocks=False):
    """Lay the four rows out to fit the terminal.

    This has to be recomputed every frame, not once at startup: if a single row
    is wider than the window the terminal wraps it onto a second physical line,
    the cursor-up at the next repaint lands in the wrong place, and the panel
    walks down the screen leaving a copy of itself behind on every frame.
    """
    w = max(width - 1, 24)      # last column left free: some terminals wrap on it

    bw = min(max(w - 46, 8), 22)
    row_sig = ("  signal   [%s] %6.1f dB  base %6.1f  thr %6.1f"
               % (bar(smooth, base.base, base.thr + 15, bw), smooth,
                  base.base, base.thr))

    half = min(max((w - 26) // 2, 4), 16)
    row_vel = ("  velocity <%s>  %s%+5.2f m/s%s"
               % (vmeter(vshow, vmax, half), BOLD, vshow, OFF))

    # Four fixed rows plus the graph; keep a couple of lines spare so the
    # panel never outgrows a short window and starts scrolling.
    rows = min(max((height - 8) // 2, 1), 4)
    graph = history_graph(hist, vmax, max(w - 11, 8), rows, blocks)

    row_sta = "  state    %s" % state
    if w >= 62:
        row_sta += "  carrier %+5.1f dB  %.0f Hz" % (carrier_d, f0)

    return [row_sig, row_vel] + graph + [row_sta]


def bar(value: float, lo: float, hi: float, width: int = 22) -> str:
    frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    frac = min(max(frac, 0.0), 1.0)
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


# -------------------------------------------------------------------- core

class DopplerSensor:
    """Full-duplex stream: emits the carrier and analyses what comes back."""

    def __init__(self, fs, f0, nfft, hop, amp, band, device=None):
        self.fs = fs
        self.f0 = f0                     # mutable: --auto-freq sweeps it
        self.nfft = nfft
        self.hop = hop
        self.amp = amp
        self.band_lo, self.band_hi = band
        self.device = device

        self.window = blackman_harris(nfft)
        # Amplitude normalisation: a full-scale sine reads 0 dBFS.
        self.norm = 2.0 / self.window.sum()
        self.buf = np.zeros(nfft, dtype=np.float64)
        self.filled = 0
        self.over = 2.5           # noise-floor over-subtraction factor
        self.floor_up = None      # per-bin noise floor, upper sideband
        self.floor_dn = None      # ditto, lower sideband
        self.blocks: deque = deque(maxlen=128)
        self.lock = threading.Lock()
        self.phase = 0.0
        self.xruns = 0
        self.stream = None

    # ---- audio callback (real time: nothing heavy in here) ---------------
    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.xruns += 1

        inc = 2.0 * np.pi * self.f0 / self.fs
        ph = self.phase + inc * np.arange(1, frames + 1)
        self.phase = float(ph[-1] % (2.0 * np.pi))
        outdata[:] = (self.amp * np.sin(ph)).astype(np.float32)[:, None]

        with self.lock:
            self.blocks.append(indata[:, 0].copy())

    def open(self):
        self.stream = sd.Stream(
            samplerate=self.fs, blocksize=self.hop, dtype="float32",
            channels=(1, 2), device=self.device, callback=self._callback,
            latency="low",
        )
        self.stream.start()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.abort()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    # ---- data flow -------------------------------------------------------
    def pop_block(self):
        with self.lock:
            return self.blocks.popleft() if self.blocks else None

    def flush(self):
        with self.lock:
            self.blocks.clear()

    def push(self, block: np.ndarray):
        n = len(block)
        self.buf[:-n] = self.buf[n:]
        self.buf[-n:] = block
        self.filled = min(self.filled + n, self.nfft)

    @property
    def ready(self) -> bool:
        return self.filled >= self.nfft

    def next_frame(self, timeout=1.0):
        """Block until a new audio block arrives and the buffer is full."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            b = self.pop_block()
            if b is None:
                time.sleep(0.002)
                continue
            self.push(b)
            if self.ready:
                return True
        return False

    # ---- analysis --------------------------------------------------------
    def analyze(self) -> dict:
        spec = np.fft.rfft(self.buf * self.window) * self.norm
        power = spec.real ** 2 + spec.imag ** 2
        binhz = self.fs / self.nfft
        c = int(round(self.f0 / binhz))

        # The carrier spans a few bins: the window, plus the clock drift
        # between the DAC and the ADC. Guard bins are excluded on both sides.
        guard = 3
        carrier = power[c - guard: c + guard + 1].sum()

        lo = max(guard + 1, int(np.ceil(self.band_lo / binhz)))
        hi = int(np.floor(self.band_hi / binhz))
        up = power[c + lo: c + hi + 1]
        dn = power[c - hi: c - lo + 1][::-1]      # flipped: offset increasing

        p_up, p_dn = up.sum(), dn.sum()
        score = db((p_up + p_dn) / max(carrier, 1e-30))

        # Velocity: SIGNED centroid of the shift, after subtracting the noise
        # floor bin by bin. Without that subtraction the centroid of pure noise
        # reads a random half a metre per second that means nothing; with it,
        # a still room pins to zero, and the sign tells approaching (upper
        # sideband) from receding (lower sideband).
        if self.floor_up is None:
            ex_up, ex_dn, floor_e = up, dn, 0.0
        else:
            # Over-subtraction: taking away the MEAN floor still leaves half
            # the bins above it by chance, and that random residue produces an
            # erratic centroid. Subtracting k times the floor makes a
            # motionless frame read a genuine zero.
            ex_up = np.maximum(up - self.over * self.floor_up, 0.0)
            ex_dn = np.maximum(dn - self.over * self.floor_dn, 0.0)
            floor_e = float(self.floor_up.sum() + self.floor_dn.sum())

        offsets = np.arange(lo, hi + 1) * binhz
        den = ex_up.sum() + ex_dn.sum()
        # Gate: below a fraction of the floor there is no echo left to measure.
        if den > 1e-30 and den > 0.10 * floor_e:
            doppler_hz = float(((ex_up - ex_dn) * offsets).sum() / den)
        else:
            doppler_hz = 0.0

        return {
            "score": score,
            "carrier_db": db(carrier),
            "speed": doppler_hz * C_SOUND / (2.0 * self.f0),
            "approaching": bool(p_up >= p_dn),
            "up": up, "dn": dn,
        }

    def learn_floor(self, r, alpha):
        """Update the per-bin noise floor from a quiet frame."""
        if self.floor_up is None:
            self.floor_up, self.floor_dn = r["up"].copy(), r["dn"].copy()
        else:
            self.floor_up += alpha * (r["up"] - self.floor_up)
            self.floor_dn += alpha * (r["dn"] - self.floor_dn)


class Baseline:
    """Baseline and threshold over a sliding window of quiet frames.

    An exponential average would be far too slow at startup and far too eager
    with something parked in front of the screen. A running median over the
    last few seconds without detections shrugs off spikes and still tracks the
    slow drift of the AGC.
    """

    def __init__(self, window_s, rate_hz, margin_db, max_db):
        self.hist = deque(maxlen=max(32, int(window_s * rate_hz)))
        self.margin = margin_db
        self.max_db = max_db
        self.base = 0.0
        self.thr = 0.0

    def add(self, score):
        self.hist.append(score)

    def recompute(self):
        if not self.hist:
            return
        h = np.fromiter(self.hist, dtype=float)
        self.base = float(np.median(h))
        # Robust deviation (MAD). High percentiles or the standard deviation
        # send the threshold through the roof the moment a single movement
        # slips into the calibration window, and then nothing is ever detected.
        sigma = 1.4826 * float(np.median(np.abs(h - self.base)))
        self.thr = self.base + min(max(3.5 * sigma, self.margin), self.max_db)


# -------------------------------------------------------------------- modes

def progress(frac, width=16):
    n = int(round(min(max(frac, 0.0), 1.0) * width))
    return CYAN + "█" * n + OFF + DIM + "░" * (width - n) + OFF


class Step:
    """One startup step: a spinner while it runs, a tick when it is done.

    Redraws are throttled, since the analysis loop calls tick() about fifty
    times a second and a spinner going that fast just looks like noise.
    """

    def __init__(self, label, tty):
        self.label, self.tty = label, tty
        self.i, self.last = 0, 0.0
        if tty:
            sys.stdout.write("  %s%s%s  %s" % (CYAN, SPIN[0], OFF, label))
            sys.stdout.flush()
        else:
            print("%s..." % label, flush=True)

    def tick(self, suffix=""):
        now = time.time()
        if not self.tty or now - self.last < 0.08:
            return
        self.last = now
        self.i += 1
        sys.stdout.write("\r\033[2K  %s%s%s  %s%s"
                         % (CYAN, SPIN[self.i % len(SPIN)], OFF,
                            self.label, suffix))
        sys.stdout.flush()

    def done(self, detail="", label=None, clear_above=0):
        """Finish the step, optionally renaming it and wiping lines above.

        A step announces what it is doing while it runs and what it achieved
        once it is done, and those are rarely the same sentence. clear_above
        lets an instruction that only applied during the step vanish with it,
        instead of sitting there afterwards telling the user to hold still.
        """
        line = "  %s✔%s  %s%s" % (GREEN, OFF,
                                    self.label if label is None else label,
                                    detail)
        if self.tty:
            out = "\r\033[2K" + "\033[1A\033[2K" * clear_above
            sys.stdout.write(out + line + "\n")
            sys.stdout.flush()
        else:
            print(line, flush=True)


def note(text, colour=None):
    # Resolved here, not in the signature: a default argument is bound when
    # the function is defined, which is before disable_colour() can empty it.
    if colour is None:
        colour = GREY
    for ln in text.split("\n"):
        print("     %s%s%s" % (colour, ln, OFF))


def auto_freq(sensor, candidates, step=None, verbose=False) -> float:
    """Sweep carriers and keep the one that comes back strongest.

    Laptop speakers and microphones fall off a cliff above ~20 kHz and the
    exact response varies by model, so it is worth measuring rather than
    guessing.
    """
    best, best_level = candidates[0], -np.inf
    table = []
    for f in candidates:
        sensor.f0 = f
        sensor.flush()
        t_end = time.time() + 0.8
        levels = []
        while time.time() < t_end:
            if not sensor.next_frame():
                break
            if step is not None:
                step.tick()
            if time.time() > t_end - 0.4:          # skip the transient
                levels.append(sensor.analyze()["carrier_db"])
        level = float(np.median(levels)) if levels else -999.0
        if level > best_level:
            best, best_level = f, level
        table.append((f, level))
    sensor.f0 = best
    if verbose:
        print()
        for f, level in table:
            print("     %6.0f Hz : carrier %6.1f dBFS%s"
                  % (f, level, "   <-- best" if f == best else ""))
    return best


def run(args):
    sensor = DopplerSensor(
        fs=args.samplerate, f0=args.freq, nfft=args.nfft, hop=args.hop,
        amp=args.amp, band=(args.band_lo, args.band_hi), device=args.device,
    )
    sensor.open()
    binhz = args.samplerate / args.nfft
    rate_hz = args.samplerate / args.hop
    csv = open(args.csv, "w") if args.csv else None
    tty = sys.stdout.isatty()

    try:
        if tty:
            print("\n  %sdoppler radar%s   %sacoustic motion sensing%s\n"
                  % (BOLD, OFF, DIM, OFF))

        st = Step("Warming up the audio stream", tty)
        t_end = time.time() + 1.0
        while time.time() < t_end:
            sensor.next_frame()
            st.tick()
        st.done(label="Audio stream ready")

        if args.auto_freq:
            cands = [f for f in range(17000, 21001, 500)
                     if f < args.samplerate / 2 - 1200]
            st = Step("Finding the best frequency", tty)
            best = auto_freq(sensor, cands, st, args.verbose)
            st.done(":  %s%.0f Hz%s" % (BOLD, best, OFF),
                label="Best frequency")
        else:
            print("  %s\u2714%s  Frequency:  %s%.0f Hz%s"
                  % (GREEN, OFF, BOLD, sensor.f0, OFF))
            note("--auto-freq picks whichever one your hardware handles best")

        total = args.warmup + args.calib
        print()
        note("Hold still and keep clear of the screen.", BOLD)
        st = Step("Listening to the room", tty)
        base = Baseline(args.window, rate_hz, args.margin, args.max_threshold)
        carriers = []
        t_room = time.time()
        while True:
            el = time.time() - t_room
            if el >= total:
                break
            if not sensor.next_frame():
                print()
                note("No audio from the microphone. Is permission granted?",
                     YELLOW)
                return 1
            r = sensor.analyze()
            # The opening stretch is thrown away on purpose: the built-in
            # microphone runs automatic gain control and takes seconds to
            # settle, and measuring before that yields a lying baseline.
            if el >= args.warmup:
                base.add(r["score"])
                sensor.learn_floor(r, 0.15)
                carriers.append(r["carrier_db"])
            st.tick("  %s  %2ds" % (progress(el / total), int(total - el) + 1))
        base.recompute()
        carrier_ref = float(np.median(carriers))
        st.done(label="Room calibrated", clear_above=1)

        if carrier_ref < -75:
            print("\n  %s\u2717%s  %sThe tone is barely coming back.%s"
                  % (YELLOW, OFF, BOLD, OFF))
            note("Turn the volume up and use the built-in speakers and\n"
                 "microphone. Then try --auto-freq, or --freq 17000.", YELLOW)
        if args.verbose:
            note("baseline %.1f dB   threshold %.1f dB   carrier %.1f dBFS"
                 % (base.base, base.thr, carrier_ref))
            note("%.1f Hz/bin   %.0f ms window   Doppler band %.0f-%.0f Hz "
                 "(%.2f-%.2f m/s)"
                 % (binhz, 1000 * args.nfft / args.samplerate,
                    args.band_lo, args.band_hi,
                    args.band_lo * C_SOUND / (2 * sensor.f0),
                    args.band_hi * C_SOUND / (2 * sensor.f0)))

        print("\n  %s\u25b8%s  Move something in front of the screen."
              "   %sCtrl-C to quit.%s\n" % (CYAN, OFF, DIM, OFF))

        if csv:
            csv.write("t,score_db,threshold_db,carrier_db,speed_ms,detected\n")

        print("  %s   receding <-------- velocity --------> approaching%s"
              % (DIM, OFF))
        t0 = time.time()
        last_log = 0.0
        last_hit = -1e9
        streak = 0
        smooth = base.base
        vel = 0.0
        # Two samples per column in braille, so the buffer holds twice what
        # the widest sensible terminal can draw.
        hist = deque([0.0] * 480, maxlen=480)
        drawn = 0
        if tty:
            # Autowrap off while the panel is live: if the window is ever too
            # narrow the terminal clips the row instead of wrapping it, which
            # would throw the cursor arithmetic off by a line.
            sys.stdout.write("\033[?7l\033[?25l")
        i = 0
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            if not sensor.next_frame():
                continue
            r = sensor.analyze()
            i += 1

            smooth = 0.6 * smooth + 0.4 * r["score"]     # anti-flicker
            now = time.time()
            if smooth > base.thr:
                streak += 1
                if streak >= args.debounce:     # debounce: ignore lone spikes
                    last_hit = now
            else:
                streak = 0
                base.add(r["score"])            # only learn while quiet
                sensor.learn_floor(r, 0.02)
            if i % 16 == 0:
                base.recompute()

            active = (now - last_hit) < args.hold
            vel = 0.7 * vel + 0.3 * r["speed"]        # velocity jitters too
            # The estimator already silences itself when the room is still, so
            # it is always on display: even sub-threshold motion shows up.
            vshow = 0.0 if abs(vel) < 0.005 else vel      # avoid "-0.00 m/s"
            hist.append(vshow)

            if active:
                state = (GREEN + "* OBJECT  " + OFF +
                         ("approaching" if vshow >= 0 else "receding   "))
            else:
                state = GREY + "  no motion          " + OFF

            if tty:
                cols, rows_t = term_size()
                lines = build_panel(cols, rows_t, smooth, base, vshow,
                                    args.vmax, hist, state,
                                    r["carrier_db"] - carrier_ref, sensor.f0,
                                    args.blocks)
                if drawn == 0:
                    sys.stdout.write("\n" * len(lines))
                    drawn = len(lines)
                sys.stdout.write("\033[%dA" % drawn)
                for ln in lines:
                    sys.stdout.write("\033[2K" + ln + "\n")
                # A resize can shrink the panel: wipe the rows it no longer
                # uses and step back onto its new last line, or the cursor
                # arithmetic drifts from here on.
                extra = drawn - len(lines)
                if extra > 0:
                    sys.stdout.write("\033[2K\n" * extra)
                    sys.stdout.write("\033[%dA" % extra)
                drawn = len(lines)
                sys.stdout.flush()
            elif now - last_log > 0.5:
                # Sin terminal (salida a fichero o a una tubería) el panel no
                # tiene sentido: los códigos de escape lo llenarían de basura.
                last_log = now
                print("%6.1fs  %7.2f dB  thr %7.2f  %+5.2f m/s  %s"
                      % (now - t0, smooth, base.thr, vshow,
                         "OBJECT" if active else "-"), flush=True)

            if csv:
                csv.write("%.3f,%.2f,%.2f,%.2f,%.3f,%d\n" % (
                    now - t0, r["score"], base.thr, r["carrier_db"],
                    r["speed"], int(active)))

    except KeyboardInterrupt:
        pass
    finally:
        if tty:
            sys.stdout.write("\033[?7h\033[?25h")     # autowrap and cursor back
            sys.stdout.flush()
        sensor.close()
        if csv:
            csv.close()
            print("\nMeasurements written to %s" % args.csv)
        if sensor.xruns:
            print("(%d audio buffer glitches)" % sensor.xruns)
    print()
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Motion detector using acoustic Doppler shift (POC).")
    p.add_argument("--list-devices", action="store_true",
                   help="list the audio devices and exit")
    p.add_argument("--device", default=None,
                   help="device as 'input,output' (indices or names)")
    p.add_argument("--freq", type=float, default=19000.0,
                   help="carrier frequency in Hz (default 19000)")
    p.add_argument("--auto-freq", action="store_true",
                   help="sweep 17-21 kHz and pick the best-received carrier")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--nfft", type=int, default=8192,
                   help="FFT size (8192 => 5.9 Hz/bin, 170 ms window)")
    p.add_argument("--hop", type=int, default=1024,
                   help="samples between analyses (1024 => 21 ms refresh)")
    p.add_argument("--amp", type=float, default=0.15, help="tone amplitude, 0-1")
    p.add_argument("--band-lo", type=float, default=15.0,
                   help="minimum Doppler offset in Hz")
    p.add_argument("--band-hi", type=float, default=600.0,
                   help="maximum Doppler offset in Hz")
    p.add_argument("--warmup", type=float, default=6.0,
                   help="seconds to let the microphone AGC settle")
    p.add_argument("--calib", type=float, default=3.0,
                   help="seconds of empty-room calibration")
    p.add_argument("--window", type=float, default=12.0,
                   help="seconds of history for the sliding baseline")
    p.add_argument("--margin", type=float, default=6.0,
                   help="minimum threshold margin over the baseline, in dB")
    p.add_argument("--max-threshold", type=float, default=14.0,
                   help="maximum threshold margin over the baseline, in dB")
    p.add_argument("--debounce", type=int, default=2,
                   help="consecutive frames over threshold to declare a hit")
    p.add_argument("--hold", type=float, default=0.5,
                   help="seconds a detection is held after the last peak")
    p.add_argument("--vmax", type=float, default=1.0,
                   help="full scale of the velocity meter, in m/s")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="stop automatically after N seconds (0 = no limit)")
    p.add_argument("--csv", default=None, help="dump the measurements to a CSV")
    p.add_argument("--blocks", action="store_true",
                   help="draw the history in chunky blocks instead of braille")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show the numbers behind the calibration")
    args = p.parse_args()

    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        disable_colour()

    if args.list_devices:
        print(sd.query_devices())
        return 0
    if args.device:
        parts = [x.strip() for x in args.device.split(",")]
        args.device = tuple(int(x) if x.lstrip("-").isdigit() else x for x in parts)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
