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


def vmeter(v: float, vmax: float, half: int = 16) -> str:
    """Bidirectional meter: zero at the centre, approaching to the right."""
    frac = min(max(v / vmax, -1.0), 1.0)
    n = int(round(abs(frac) * half))
    if v >= 0:
        left, right = "·" * half, GREEN + "█" * n + OFF + "·" * (half - n)
    else:
        left, right = "·" * (half - n) + CYAN + "█" * n + OFF, "·" * half
    return left + BOLD + "│" + OFF + right


def spark(v: float, vmax: float) -> str:
    """One character, height proportional to |v|, coloured by direction."""
    lvl = int(round(min(abs(v) / vmax, 1.0) * 8))
    if lvl == 0:
        return GREY + "·" + OFF
    return (GREEN if v >= 0 else CYAN) + BLOCKS[lvl] + OFF


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

def warmup(sensor, seconds, msg):
    """Fill the buffer and let the microphone AGC settle."""
    print(msg, end="", flush=True)
    t_end = time.time() + seconds
    while time.time() < t_end:
        sensor.next_frame()
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.25)
    print(" done")


def auto_freq(sensor, candidates) -> float:
    """Sweep carriers and keep the one that comes back strongest.

    Laptop speakers and microphones fall off a cliff above ~20 kHz and the
    exact response varies by model, so it is worth measuring rather than
    guessing.
    """
    print("Scanning for the best carrier (~%.0f s, stay still):"
          % (0.8 * len(candidates)))
    best, best_level = candidates[0], -np.inf
    for f in candidates:
        sensor.f0 = f
        sensor.flush()
        t_end = time.time() + 0.8
        levels = []
        while time.time() < t_end:
            if not sensor.next_frame():
                break
            if time.time() > t_end - 0.4:          # skip the transient
                levels.append(sensor.analyze()["carrier_db"])
        level = float(np.median(levels)) if levels else -999.0
        mark = ""
        if level > best_level:
            best, best_level, mark = f, level, "   <-- best"
        print("   %6.0f Hz : carrier %6.1f dBFS%s" % (f, level, mark))
    sensor.f0 = best
    print("Carrier chosen: %.0f Hz\n" % best)
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

    try:
        warmup(sensor, 1.0, "Opening the stream")

        if args.auto_freq:
            cands = [f for f in range(17000, 21001, 500)
                     if f < args.samplerate / 2 - 1200]
            auto_freq(sensor, cands)

        print("Carrier %.0f Hz  |  %.1f Hz/bin  |  %.0f ms window  |  "
              "Doppler band %.0f-%.0f Hz (%.2f-%.2f m/s)"
              % (sensor.f0, binhz, 1000 * args.nfft / args.samplerate,
                 args.band_lo, args.band_hi,
                 args.band_lo * C_SOUND / (2 * sensor.f0),
                 args.band_hi * C_SOUND / (2 * sensor.f0)))

        # The built-in microphone runs automatic gain control and takes a few
        # seconds to settle; measuring before that yields a lying baseline.
        warmup(sensor, args.warmup,
               "Settling the microphone AGC (%.0f s)" % args.warmup)

        base = Baseline(args.window, rate_hz, args.margin, args.max_threshold)
        print("Calibrating for %.1f s: keep clear of the screen" % args.calib,
              end="", flush=True)
        carriers = []
        t_end = time.time() + args.calib
        while time.time() < t_end:
            if not sensor.next_frame():
                print("\nNo audio arriving from the microphone. "
                      "Is permission granted?")
                return 1
            r = sensor.analyze()
            base.add(r["score"])
            sensor.learn_floor(r, 0.15)
            carriers.append(r["carrier_db"])
        base.recompute()
        carrier_ref = float(np.median(carriers))
        print("\n   baseline %.1f dB   threshold %.1f dB   carrier %.1f dBFS"
              % (base.base, base.thr, carrier_ref))
        if carrier_ref < -75:
            print("   WARNING: the carrier is barely coming back. Turn the volume\n"
                  "   up, use the BUILT-IN speakers and microphone (no Bluetooth),\n"
                  "   and try --auto-freq or a lower carrier (--freq 17000).")
        print("\nMove something in front of the screen.  Ctrl-C to quit.\n")

        if csv:
            csv.write("t,score_db,threshold_db,carrier_db,speed_ms,detected\n")

        print("   receding ◄──────── velocity ────────► approaching")
        t0 = time.time()
        last_hit = -1e9
        streak = 0
        smooth = base.base
        vel = 0.0
        hist = deque([0.0] * 56, maxlen=56)
        sys.stdout.write("\n" * 4)        # room for the four-line panel
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
                state = (GREEN + "● OBJECT  " + OFF +
                         ("approaching" if vshow >= 0 else "receding   "))
            else:
                state = GREY + "○ no motion          " + OFF

            lines = [
                "  signal    [%s] %6.1f dB      baseline %6.1f   threshold %6.1f"
                % (bar(smooth, base.base, base.thr + 15), smooth,
                   base.base, base.thr),
                "  velocity ◄%s►  %s%+5.2f m/s%s"
                % (vmeter(vshow, args.vmax), BOLD, vshow, OFF),
                "  history   %s" % "".join(spark(v, args.vmax) for v in hist),
                "  state     %s   carrier %+5.1f dB   %.0f Hz"
                % (state, r["carrier_db"] - carrier_ref, sensor.f0),
            ]
            sys.stdout.write("\033[%dA" % len(lines))
            for ln in lines:
                sys.stdout.write("\033[2K" + ln + "\n")
            sys.stdout.flush()

            if csv:
                csv.write("%.3f,%.2f,%.2f,%.2f,%.3f,%d\n" % (
                    now - t0, r["score"], base.thr, r["carrier_db"],
                    r["speed"], int(active)))

    except KeyboardInterrupt:
        pass
    finally:
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
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0
    if args.device:
        parts = [x.strip() for x in args.device.split(",")]
        args.device = tuple(int(x) if x.lstrip("-").isdigit() else x for x in parts)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
