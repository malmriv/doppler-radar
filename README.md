# Detecting motion in front of the screen with the Doppler effect

A friend sent me a video of something like this working: an ordinary laptop, no
extra hardware, sensing a hand through the speaker and the microphone. My
reaction was **"there is no way that works"**.

So I vibecoded it one afternoon to get it out of my system, not believing it for
a second and with no intention of this going anywhere. To my surprise, it works.

![demo](demo.gif)

## The physics, briefly

The speaker emits a continuous tone at ~20 kHz, inaudible to just about any
adult. The microphone picks it up over the direct path — the **carrier**,
enormous and perfectly steady — plus every echo in the room. Anything standing
still reflects at the same frequency. Anything moving sends the echo back
shifted:

$$\Delta f = \frac{2vf_0}{c}$$

An object moving at 30 cm/s on a 20 kHz carrier gives about 35 Hz. In relative
terms that is a rounding error, 0.17 %; but next to a spectral line as clean as
the carrier it is six FFT bins, and you can see it perfectly well.

The program measures the energy that shows up on either side of the carrier,
divides it by the carrier power, and compares the result against the baseline of
the empty room. That division is what makes the whole thing immune to the
microphone's automatic gain control: when the system raises or lowers the input
gain, carrier and sidebands rise and fall together and the ratio never notices.

Whether the echo comes back above or below the carrier tells you whether the
object is approaching or receding. That is all the physics there is here.

## How to make this work on your computer

```bash
git clone https://github.com/malmriv/doppler.git
cd doppler
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python doppler_sense.py --auto-freq
```

Startup takes about 15 seconds: it finds the best carrier, waits for the
microphone to settle, and calibrates the room. **Keep clear of the screen while
it calibrates.** Then move something in front of it. `Ctrl-C` to quit.

Now the conditions, which matter more than the code does:

- **Built-in speakers, with the volume up.** Halfway is plenty. Muted means
  there is nothing to detect.
- **No Bluetooth, no headphones.** AirPods and the like neither reach 20 kHz nor
  transmit uncompressed, and with headphones on the tone never reaches the air.
  Speaker and microphone both **built-in**.
- **The microphone has to stay put.** Move the laptop and the entire geometry
  moves with it, so everything else appears to move too. On a desk it works; on
  your lap it does not.
- **Grant microphone permission** when the system asks.
- **The room does not need to be quiet.** Ordinary noise — voices, keyboards,
  fans — lives below 8 kHz; at 20 kHz there is 37 dB less of it. It is the
  quietest corner of the spectrum, which is exactly why the tone is ultrasonic
  rather than audible.
- **Mind who else is moving around.** A continuous tone carries no range
  information: it cannot tell your hand from someone walking past behind you.

Tested on a MacBook Pro. Linux and Windows should behave the same
(`sounddevice` wraps PortAudio on all three), but I have not verified it.

### If it does not work

| Symptom | Usual cause |
|---|---|
| Warns that the carrier is barely coming back | Volume too low, output routed over Bluetooth, or headphones plugged in |
| Detects nothing however much you move | You calibrated with movement in front of it; restart and keep clear |
| Detects constantly | Something nearby is moving: a fan, a curtain, someone walking past |
| Nothing at all, no carrier and no noise | Microphone permission missing |

Useful options: `--freq 17000` if your hardware does not reach that high,
`--margin` to make it more or less sensitive, `--csv out.csv` to dump the
measurements and study them at leisure.

## What you see

```
  signal    [██████████░░░░░░░░░░░░]  -28.9 dB   baseline -42.1  threshold -35.5
  velocity ◄···········█████│················►  -0.29 m/s
  history   ····▁▁▁······▁▁▂▂▃▃▄▄▅▅▅▄▄▃▃▂▂▂▁·▁▁▂▂▂
  state     ● OBJECT  receding      carrier -0.1 dB   20000 Hz
```

Green and to the right, approaching; cyan and to the left, receding. The history
row covers the last second or so.

## But does it actually work?

That was my question too. Measured on a MacBook Pro:

- The carrier comes back at **-19 dBFS** at 21 kHz. Plenty of headroom.
- With the room still, the baseline sits at **-42 dB** with a robust deviation
  of 2.4 dB, and the threshold lands 6-8 dB above it. The detector responds to
  echoes as faint as **-40 dB** relative to the carrier.
- In 25 seconds of supposedly empty room it flagged four clean episodes, half a
  second to a second and a half each, at around 0.43 m/s. Those were not false
  positives: that was me, moving.

## What it does not do

Doppler detects **motion, not presence**. Something perfectly still in front of
the screen produces no sidebands and the program does not see it. It does
perturb the carrier amplitude a little, shown as a secondary metric, but that is
far less reliable.

It also measures no distance, so it cannot separate your hand from whatever is
happening across the room. That would take an FMCW chirp instead of a fixed
tone, which gives range and velocity at once. I did not get that far: this was
one afternoon spent proving to myself that it would not work.

## Licence

MIT. Do as you please with it.
