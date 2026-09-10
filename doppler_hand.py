#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POC de sonar Doppler acústico para macOS.

Idea: el altavoz emite un tono continuo casi inaudible (~19 kHz). El micrófono
capta ese tono por camino directo (la portadora: fortísima y perfectamente
estable) más los ecos del entorno. Todo lo que está quieto refleja en la misma
frecuencia; lo que se mueve devuelve el eco desplazado:

    Δf = 2 · v · f0 / c        (c ≈ 343 m/s)

Una mano a 30 cm/s delante de la pantalla produce Δf ≈ 33 Hz sobre 19 kHz. Es
un desplazamiento minúsculo en relativo (0.17 %), pero perfectamente visible en
el espectro porque la portadora es una raya espectral limpísima.

El detector, entonces: FFT con una ventana de lóbulos laterales muy bajos
(Blackman-Harris), medir la energía en las bandas laterales alrededor de la
portadora, normalizarla por la potencia de la portadora, y compararla con la
línea base del entorno vacío.

Esa normalización es la que hace que el invento sobreviva al AGC del micro
interno del Mac: si el sistema sube o baja la ganancia, portadora y bandas
laterales se mueven juntas y el cociente no se entera.

Limitación honesta de la física: el Doppler detecta MOVIMIENTO, no presencia.
Una mano perfectamente inmóvil no genera bandas laterales. Sí altera un poco la
amplitud de la portadora, y eso se muestra como métrica secundaria, pero es
bastante menos fiable.

Uso:
    python doppler_hand.py                 # calibra y detecta
    python doppler_hand.py --auto-freq     # busca antes la mejor portadora
    python doppler_hand.py --list-devices
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


# ---------------------------------------------------------------- utilidades

def blackman_harris(n: int) -> np.ndarray:
    """Ventana de 4 términos: lóbulos laterales a -92 dB.

    Imprescindible aquí. Con una Hann normal (-31 dB) la fuga espectral de la
    portadora enterraría las bandas laterales que queremos medir.
    """
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    w = 2 * np.pi * np.arange(n) / (n - 1)
    return a[0] - a[1] * np.cos(w) + a[2] * np.cos(2 * w) - a[3] * np.cos(3 * w)


def db(x: float) -> float:
    return 10.0 * np.log10(max(float(x), 1e-30))


BLOCKS = " ▁▂▃▄▅▆▇█"
GREEN, CYAN, GREY, BOLD, OFF = "\033[92m", "\033[96m", "\033[90m", "\033[1m", "\033[0m"


def vmeter(v: float, vmax: float, half: int = 16) -> str:
    """Medidor bidireccional: cero en el centro, acercarse hacia la derecha."""
    frac = min(max(v / vmax, -1.0), 1.0)
    n = int(round(abs(frac) * half))
    if v >= 0:
        left, right = "·" * half, GREEN + "█" * n + OFF + "·" * (half - n)
    else:
        left, right = "·" * (half - n) + CYAN + "█" * n + OFF, "·" * half
    return left + BOLD + "│" + OFF + right


def spark(v: float, vmax: float) -> str:
    """Un carácter de altura proporcional a |v|, coloreado por sentido."""
    lvl = int(round(min(abs(v) / vmax, 1.0) * 8))
    if lvl == 0:
        return GREY + "·" + OFF
    return (GREEN if v >= 0 else CYAN) + BLOCKS[lvl] + OFF


def bar(value: float, lo: float, hi: float, width: int = 22) -> str:
    frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
    frac = min(max(frac, 0.0), 1.0)
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


# ------------------------------------------------------------------- núcleo

class DopplerSensor:
    """Stream full-duplex: emite la portadora y analiza lo que vuelve."""

    def __init__(self, fs, f0, nfft, hop, amp, band, device=None):
        self.fs = fs
        self.f0 = f0                     # mutable: --auto-freq lo va cambiando
        self.nfft = nfft
        self.hop = hop
        self.amp = amp
        self.band_lo, self.band_hi = band
        self.device = device

        self.window = blackman_harris(nfft)
        # Normalización de amplitud: un seno a fondo de escala da 0 dBFS.
        self.norm = 2.0 / self.window.sum()
        self.buf = np.zeros(nfft, dtype=np.float64)
        self.filled = 0
        self.over = 2.5           # factor de sobre-resta del suelo de ruido
        self.floor_up = None      # suelo de ruido por bin, lado superior
        self.floor_dn = None      # ídem, lado inferior
        self.blocks: deque = deque(maxlen=128)
        self.lock = threading.Lock()
        self.phase = 0.0
        self.xruns = 0
        self.stream = None

    # ---- callback de audio (tiempo real: nada pesado aquí dentro) --------
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

    # ---- flujo de datos --------------------------------------------------
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
        """Bloquea hasta tener un bloque nuevo y el buffer lleno."""
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

    # ---- análisis --------------------------------------------------------
    def analyze(self) -> dict:
        spec = np.fft.rfft(self.buf * self.window) * self.norm
        power = spec.real ** 2 + spec.imag ** 2
        binhz = self.fs / self.nfft
        c = int(round(self.f0 / binhz))

        # La portadora ocupa unos pocos bins: ventana, más la deriva de reloj
        # entre el DAC y el ADC. Se excluye una guarda a ambos lados.
        guard = 3
        carrier = power[c - guard: c + guard + 1].sum()

        lo = max(guard + 1, int(np.ceil(self.band_lo / binhz)))
        hi = int(np.floor(self.band_hi / binhz))
        up = power[c + lo: c + hi + 1]
        dn = power[c - hi: c - lo + 1][::-1]      # invertido: offset creciente

        p_up, p_dn = up.sum(), dn.sum()
        score = db((p_up + p_dn) / max(carrier, 1e-30))

        # Velocidad: centroide CON SIGNO del desplazamiento, tras restarle el
        # suelo de ruido bin a bin. Sin esa resta, en reposo el centroide del
        # puro ruido da un valor aleatorio de medio metro por segundo que no
        # significa nada; con ella, en reposo se queda pegado a cero y el signo
        # distingue acercarse (banda superior) de alejarse (banda inferior).
        if self.floor_up is None:
            ex_up, ex_dn, floor_e = up, dn, 0.0
        else:
            # Sobre-resta: restar el suelo MEDIO deja la mitad de los bins por
            # encima solo por azar, y ese residuo aleatorio produce un centroide
            # errático. Restando k veces el suelo, un fotograma sin movimiento
            # se queda en cero de verdad.
            ex_up = np.maximum(up - self.over * self.floor_up, 0.0)
            ex_dn = np.maximum(dn - self.over * self.floor_dn, 0.0)
            floor_e = float(self.floor_up.sum() + self.floor_dn.sum())

        offsets = np.arange(lo, hi + 1) * binhz
        den = ex_up.sum() + ex_dn.sum()
        # Puerta: por debajo de una fracción del suelo, no hay eco que medir.
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
        """Actualiza el suelo de ruido por bin con un fotograma en calma."""
        if self.floor_up is None:
            self.floor_up, self.floor_dn = r["up"].copy(), r["dn"].copy()
        else:
            self.floor_up += alpha * (r["up"] - self.floor_up)
            self.floor_dn += alpha * (r["dn"] - self.floor_dn)


class Baseline:
    """Línea base y umbral por ventana deslizante de fotogramas 'en calma'.

    Un promedio exponencial iría demasiado lento al arrancar y demasiado rápido
    con la mano quieta delante. Una mediana móvil sobre los últimos segundos sin
    detección es robusta a picos y sigue la deriva lenta del AGC.
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
        # Desviación robusta (MAD). Usar percentiles altos o la desviación
        # típica dispara el umbral en cuanto pasa un solo evento de movimiento
        # durante la calibración, y entonces ya no detecta nada.
        sigma = 1.4826 * float(np.median(np.abs(h - self.base)))
        self.thr = self.base + min(max(3.5 * sigma, self.margin), self.max_db)


# -------------------------------------------------------------- modos de uso

def warmup(sensor, seconds, msg):
    """Llena el buffer y deja que el AGC del micro se asiente."""
    print(msg, end="", flush=True)
    t_end = time.time() + seconds
    while time.time() < t_end:
        sensor.next_frame()
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.25)
    print(" listo")


def auto_freq(sensor, candidates) -> float:
    """Barre portadoras y elige la que mejor se recibe.

    Los altavoces y micros de los Mac caen a plomo por encima de ~20 kHz y la
    respuesta concreta varía según el modelo, así que más vale medirlo.
    """
    print("Buscando la mejor portadora (~%.0f s, no toques nada):"
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
            if time.time() > t_end - 0.4:          # descarta el transitorio
                levels.append(sensor.analyze()["carrier_db"])
        level = float(np.median(levels)) if levels else -999.0
        mark = ""
        if level > best_level:
            best, best_level, mark = f, level, "   <-- mejor"
        print("   %6.0f Hz : portadora %6.1f dBFS%s" % (f, level, mark))
    sensor.f0 = best
    print("Portadora elegida: %.0f Hz\n" % best)
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
        warmup(sensor, 1.0, "Abriendo el stream")

        if args.auto_freq:
            cands = [f for f in range(17000, 21001, 500)
                     if f < args.samplerate / 2 - 1200]
            auto_freq(sensor, cands)

        print("Portadora %.0f Hz  |  %.1f Hz/bin  |  ventana %.0f ms  |  "
              "banda Doppler %.0f-%.0f Hz (%.2f-%.2f m/s)"
              % (sensor.f0, binhz, 1000 * args.nfft / args.samplerate,
                 args.band_lo, args.band_hi,
                 args.band_lo * C_SOUND / (2 * sensor.f0),
                 args.band_hi * C_SOUND / (2 * sensor.f0)))

        # El micro interno lleva control automático de ganancia y tarda unos
        # segundos en asentarse; medir antes da una línea base mentirosa.
        warmup(sensor, args.warmup, "Estabilizando el micro (%.0f s)" % args.warmup)

        base = Baseline(args.window, rate_hz, args.margin, args.max_threshold)
        print("Calibrando %.1f s: aparta las manos de la pantalla" % args.calib,
              end="", flush=True)
        carriers = []
        t_end = time.time() + args.calib
        while time.time() < t_end:
            if not sensor.next_frame():
                print("\nNo llega audio del micrófono. ¿Permisos concedidos?")
                return 1
            r = sensor.analyze()
            base.add(r["score"])
            sensor.learn_floor(r, 0.15)
            carriers.append(r["carrier_db"])
        base.recompute()
        carrier_ref = float(np.median(carriers))
        print("\n   línea base %.1f dB   umbral %.1f dB   portadora %.1f dBFS"
              % (base.base, base.thr, carrier_ref))
        if carrier_ref < -75:
            print("   AVISO: apenas se recibe la portadora. Sube el volumen, usa\n"
                  "   altavoces y micro INTERNOS (nada de Bluetooth) y prueba con\n"
                  "   --auto-freq o una frecuencia más baja (--freq 17000).")
        print("\nMueve la mano delante de la pantalla.  Ctrl-C para salir.\n")

        if csv:
            csv.write("t,score_db,threshold_db,carrier_db,speed_ms,detected\n")

        print("   alejándose ◄──────── velocidad ────────► acercándose")
        t0 = time.time()
        last_hit = -1e9
        streak = 0
        smooth = base.base
        vel = 0.0
        hist = deque([0.0] * 56, maxlen=56)
        sys.stdout.write("\n" * 4)        # hueco para el panel de 4 líneas
        i = 0
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            if not sensor.next_frame():
                continue
            r = sensor.analyze()
            i += 1

            smooth = 0.6 * smooth + 0.4 * r["score"]     # anti-parpadeo
            now = time.time()
            if smooth > base.thr:
                streak += 1
                if streak >= args.debounce:     # antirrebote: nada de picos sueltos
                    last_hit = now
            else:
                streak = 0
                base.add(r["score"])            # solo aprende en calma
                sensor.learn_floor(r, 0.02)
            if i % 16 == 0:
                base.recompute()

            active = (now - last_hit) < args.hold
            vel = 0.7 * vel + 0.3 * r["speed"]        # la velocidad también tiembla
            # El estimador ya se silencia solo en calma, así que se muestra
            # siempre: se ve incluso el movimiento que no llega al umbral.
            vshow = vel
            hist.append(vshow)

            if active:
                estado = (GREEN + "● MANO  " + OFF +
                          ("acercándose" if vshow >= 0 else "alejándose "))
            else:
                estado = GREY + "○ sin movimiento      " + OFF

            lines = [
                "  señal     [%s] %6.1f dB      base %6.1f   umbral %6.1f"
                % (bar(smooth, base.base, base.thr + 15), smooth,
                   base.base, base.thr),
                "  veloc.  ◄%s►  %s%+5.2f m/s%s"
                % (vmeter(vshow, args.vmax), BOLD, vshow, OFF),
                "  historia  %s" % "".join(spark(v, args.vmax) for v in hist),
                "  estado    %s   portadora %+5.1f dB   %.0f Hz"
                % (estado, r["carrier_db"] - carrier_ref, sensor.f0),
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
            print("\nCSV guardado en %s" % args.csv)
        if sensor.xruns:
            print("(%d fallos de buffer de audio)" % sensor.xruns)
    print()
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Detector de mano por efecto Doppler acústico (POC).")
    p.add_argument("--list-devices", action="store_true",
                   help="lista los dispositivos de audio y sale")
    p.add_argument("--device", default=None,
                   help="dispositivo 'entrada,salida' (índices o nombres)")
    p.add_argument("--freq", type=float, default=19000.0,
                   help="frecuencia de la portadora en Hz (def. 19000)")
    p.add_argument("--auto-freq", action="store_true",
                   help="barre 17-21 kHz y elige la portadora mejor recibida")
    p.add_argument("--samplerate", type=int, default=48000)
    p.add_argument("--nfft", type=int, default=8192,
                   help="tamaño de FFT (8192 => 5.9 Hz/bin, ventana de 170 ms)")
    p.add_argument("--hop", type=int, default=1024,
                   help="muestras entre análisis (1024 => 21 ms de refresco)")
    p.add_argument("--amp", type=float, default=0.15, help="amplitud del tono, 0-1")
    p.add_argument("--band-lo", type=float, default=15.0,
                   help="offset Doppler mínimo en Hz")
    p.add_argument("--band-hi", type=float, default=600.0,
                   help="offset Doppler máximo en Hz")
    p.add_argument("--warmup", type=float, default=6.0,
                   help="segundos de espera a que se asiente el AGC del micro")
    p.add_argument("--calib", type=float, default=3.0,
                   help="segundos de calibración del entorno vacío")
    p.add_argument("--window", type=float, default=12.0,
                   help="segundos de historia para la línea base deslizante")
    p.add_argument("--margin", type=float, default=6.0,
                   help="margen mínimo del umbral sobre la línea base, en dB")
    p.add_argument("--max-threshold", type=float, default=14.0,
                   help="margen máximo del umbral sobre la línea base, en dB")
    p.add_argument("--debounce", type=int, default=2,
                   help="frames consecutivos sobre el umbral para dar detección")
    p.add_argument("--hold", type=float, default=0.5,
                   help="segundos que se sostiene la detección tras el último pico")
    p.add_argument("--vmax", type=float, default=1.0,
                   help="fondo de escala del medidor de velocidad, en m/s")
    p.add_argument("--seconds", type=float, default=0.0,
                   help="parar automáticamente tras N segundos (0 = sin límite)")
    p.add_argument("--csv", default=None, help="volcar las medidas a un CSV")
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
