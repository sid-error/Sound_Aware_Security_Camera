"""
Microphone spacing / alignment check. Prints the raw lag and angle of every loud
sound (no model, the camera does not move) for TEST_SECONDS, then a summary.

Clap ~5 times at each position, about 1-2 m from the microphones:
    1. straight in front of the mics (0 deg)
    2. far LEFT, in line with the two mics (-90 deg)
    3. far RIGHT, in line with the two mics (+90 deg)

The largest possible lag is spacing / 343 m/s * 48000:
    10 cm -> 14 samples      25 cm -> 35 samples
So the lags at far left/right show the real spacing, and the lag in front
should be ~0 (if not, the channels have a fixed delay or the mics are turned).

    python3 doa_check.py
"""

import queue
import time

import numpy as np
import sounddevice as sd

import config as cfg
from doa import DirectionFinder

TEST_SECONDS = 60
WINDOW = int(cfg.SAMPLE_RATE * 0.5)
# search wider than either spacing so the real lag is never cut off
finder = DirectionFinder(0.40, cfg.SOUND_SPEED, cfg.SAMPLE_RATE, cfg.DOA_SEGMENT // 2)
blocks = queue.Queue()
lags = []


def callback(indata, frames, time_info, status):
    blocks.put(indata.copy())


def angle(lag, spacing):
    r = np.clip(lag / cfg.SAMPLE_RATE * cfg.SOUND_SPEED / spacing, -1, 1)
    return np.degrees(np.arcsin(r))


print(f"Listening for {TEST_SECONDS} s. Clap 5x in FRONT, 5x far LEFT, 5x far RIGHT.")
buf = np.zeros((0, cfg.CHANNELS), dtype=np.float32)
end = time.time() + TEST_SECONDS
with sd.InputStream(device=cfg.AUDIO_DEVICE, samplerate=cfg.SAMPLE_RATE,
                    blocksize=cfg.BLOCK_SIZE, channels=cfg.CHANNELS,
                    dtype="float32", callback=callback):
    while time.time() < end:
        buf = np.concatenate([buf, blocks.get()])
        if len(buf) < WINDOW:
            continue
        window, buf = buf[:WINDOW], buf[WINDOW:]
        rms = np.sqrt(np.mean(window ** 2, axis=0)).max()
        if rms < cfg.RMS_GATE:
            continue
        d = finder.estimate(window)
        lags.append(d["lag"])
        print(f"[{end - time.time():4.0f}s left] rms={rms:.3f}  lag={d['lag']:+6.1f} samples  "
              f"q={d['quality']:4.1f}  angle if 10cm: {angle(d['lag'], 0.10):+6.1f}  "
              f"if 25cm: {angle(d['lag'], 0.25):+6.1f}")

if lags:
    a = np.array(lags)
    print(f"\n{len(a)} sounds. lag min {a.min():+.1f}  max {a.max():+.1f} samples")
    print("max |lag| ~14 -> mics 10 cm apart; ~35 -> 25 cm; beyond -> echoes/channel delay")
else:
    print(f"\nNo loud sounds heard (gate {cfg.RMS_GATE}).")
