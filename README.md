# Sound Sentry 🔊📷

**A suspicious-sound-triggered PTZ security camera system powered by edge AI on Raspberry Pi.**

Sound Sentry continuously listens through two microphones, classifies every sound with an on-device neural network, and — *only* when a sound is classified as **suspicious** (e.g., glass breaking, forced entry, knocking, footsteps) — estimates the direction of arrival and turns a pan/tilt IP camera to face it. Normal sounds (rain, wind, traffic, dogs, talking) are silently ignored.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Hardware Requirements](#hardware-requirements)
- [Wiring & Pin Configuration](#wiring--pin-configuration)
- [Software Requirements](#software-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Modules In Depth](#modules-in-depth)
  - [main.py — Orchestrator & Audio Loop](#mainpy--orchestrator--audio-loop)
  - [classifier.py — TFLite Inference](#classifierpy--tflite-inference)
  - [doa.py — Direction of Arrival (GCC-PHAT)](#doapy--direction-of-arrival-gcc-phat)
  - [camera.py — PTZ Camera Controller](#camerapy--ptz-camera-controller)
  - [config.py — All Settings](#configpy--all-settings)
  - [term.py — Terminal Logging](#termpy--terminal-logging)
  - [doa_check.py — Microphone Calibration Tool](#doa_checkpy--microphone-calibration-tool)
- [The AI Model](#the-ai-model)
- [Camera Control Details](#camera-control-details)
- [Direction of Arrival Estimation](#direction-of-arrival-estimation)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Sound Sentry is an **edge-AI security system** designed to run entirely on a Raspberry Pi without cloud connectivity. The system leverages transfer learning on [YAMNet](https://tfhub.dev/google/yamnet/1) (pre-trained on Google's AudioSet) with a custom dense classification head, exported as a single quantised TensorFlow Lite file.

**Key design decisions:**

| Decision | Rationale |
|---|---|
| Edge-only inference | No cloud latency; works offline; no privacy risk |
| TFLite quantisation | ~3.7 MB model, ~7–10 ms inference on Pi |
| RMS loudness gate | Prevents the model from misfiring on silence |
| Camera busy/cooldown window | Motor noise cannot trigger itself |
| GCC-PHAT over direct cross-correlation | O(N log N), robust to room reverberation |
| Continuous PTZ moves with `move=repeat` | One smooth pan vs. the old jerky 1.5 s chunk approach |
| Dead-reckoning + home-preset re-sync | Compensates motor speed variation without encoder feedback |

---

## How It Works

The processing pipeline runs **every 0.5 s** on the most recent **1 s** of stereo audio:

```
┌──────────────────────────────────────────────────────────────┐
│  Stereo audio window (48 kHz, 1 s, 2 channels)               │
└───────────────────────────┬──────────────────────────────────┘
                            │
                    ① Loudness Gate
                  max(RMS_L, RMS_R) < 0.05?
                      YES → ignore (quiet)
                            │ NO
                    ② Camera State Check
                  camera moving / in cooldown?
                      YES → ignore
                            │ NO
                    ③ AI Classification
                  P(suspicious) < 0.60?
                      YES → ignore (normal sound)
                            │ NO
                    ④ Consecutive-hit guard
                  < N windows above threshold?
                      YES → accumulate
                            │ N reached
                    ⑤ Direction of Arrival (GCC-PHAT)
                  angle = arcsin(c · Δt / d)
                            │
                    ⑥ Camera Move
                  turn to estimated angle (−90° to +90°)
                  skipped if Δ < 5° or already pointing there
```

**Step-by-step:**

1. **Loudness gate** — If the maximum per-channel RMS of the window is below `RMS_GATE = 0.05`, the window is silently discarded. This is *required* because the model produces high suspicious scores on silent or near-silent audio.

2. **Camera-busy guard** — If the camera is currently executing a move, or is within the 2 s post-move cooldown window, the audio window is ignored. This prevents the camera's own motor noise from triggering another move.

3. **Classification** — The 48 kHz stereo window is downsampled to 16 kHz mono (anti-aliased, using `resample_poly`), peak-normalised to [−1, 1], and fed to the quantised TFLite model (YAMNet + dense head) to obtain `P(suspicious)`.

4. **Consecutive-hit guard** — If `CONSECUTIVE_HITS > 1`, multiple windows in a row must all exceed the threshold before the system acts. This reduces false positives from transient artefacts.

5. **Direction of Arrival (DOA)** — GCC-PHAT cross-correlation between the left and right channels over the loudest `DOA_SEGMENT = 16384` samples of the detection window. The time delay Δt is converted to an arrival angle via `angle = arcsin(c · Δt / d)` where `c = 343 m/s` (speed of sound) and `d = 0.25 m` (mic spacing).

6. **Camera move** — The camera is commanded to pan to the estimated angle. Moves smaller than `MIN_TURN_DEGREES = 5°` are skipped. Targets within `SNAP_TO_HOME_DEGREES = 7.5°` of front use the home preset (an absolute motor position) for exact re-synchronisation.

---

## Hardware Requirements

| Component | Details |
|---|---|
| **Computer** | Raspberry Pi (64-bit Raspberry Pi OS, kernel 6.x) — Pi 4B or Pi 5 recommended |
| **Microphones** | Two **INMP441** I2S MEMS microphone modules (left + right), wired to the Pi GPIO header via a breadboard, **25 cm apart**. See [Wiring & Pin Configuration](#wiring--pin-configuration) |
| **Camera** | TP-Link **TL-SC4171G** Wireless Pan/Tilt IP camera (HTTP CGI control via `/cgi-bin/operator/ptzset`) |
| **Network** | Camera connected to the Raspberry Pi with an **Ethernet cable** (same subnet as `CAMERA_IP`) |
| **Mounting** | Camera, microphones, and Pi mounted rigidly together so the entire unit moves as one piece |

> **Important:** The camera, microphones, and Pi must be fixed relative to each other. The DOA calibration (`MIC_DISTANCE`, pan-speed values) are only valid for one physical configuration. If anything shifts, recalibrate.

**Microphone geometry:**
- Column 0 (left channel) = left microphone
- Column 1 (right channel) = right microphone
- Lag > 0 → left channel delayed → sound arrives from the **right** → camera turns **right**

**Camera CGI endpoints used:**
| Endpoint | Purpose |
|---|---|
| `GET /cgi-bin/operator/ptzset?move=right&move=repeat` | Start continuous right pan |
| `GET /cgi-bin/operator/ptzset?move=left&move=repeat` | Start continuous left pan |
| `GET /cgi-bin/operator/ptzset?move=stop` | Stop motor |
| `GET /cgi-bin/operator/ptzset?gotoserverpresetno=N` | Go to saved preset N |

---

## Wiring & Pin Configuration

The two microphones are **INMP441** MEMS microphone modules (I2S, 3.3V). All pin numbers below are **physical board pin numbers** on the Raspberry Pi 40-pin header.

**Breadboard layout:** the `SCK`, `WS` and `SD` pins of both microphones share one breadboard row each, and each of those rows is wired to a single Pi pin. `VDD` of each mic goes to a 3.3V pin and `GND` of each mic goes to a ground pin. The `L/R` pin selects the channel: one mic has `L/R` tied to its own `VDD` row, the other has it tied to its own `GND` row.

| Mic pin | Mic 1 (Left ear) | Mic 2 (Right ear) |
|---|---|---|
| `VDD` | Pi Pin 1 (3.3V) | Pi Pin 17 (3.3V) |
| `GND` | Pi Pin 6 (GND) | Pi Pin 9 (GND) |
| `L/R` | Tied to Mic 1's `GND` row → **left** channel | Tied to Mic 2's `VDD` row → **right** channel |
| `SCK` | Shared row → Pi Pin 12 | Same shared row as Mic 1 → Pi Pin 12 |
| `WS` | Shared row → Pi Pin 35 | Same shared row as Mic 1 → Pi Pin 35 |
| `SD` | Shared row → Pi Pin 38 | Same shared row as Mic 1 → Pi Pin 38 |

**Shared I2S bus connections to the Pi:**

| Signal | Pi Pin | Function |
|---|---|---|
| `SCK` | 12 | Bit clock |
| `WS` | 35 | Word select (left/right frame clock) |
| `SD` | 38 | Serial data |

`L/R` tied low puts a mic on the left channel (column 0) and tied high puts it on the right channel (column 1), matching the [microphone geometry](#hardware-requirements) convention above. If left and right turn out swapped, swap the two `L/R` ties.

### I2S overlay setup

> **TODO:** I2S overlay configuration to be added later.

### Camera connection

The PTZ camera is connected to the Raspberry Pi with an **Ethernet cable**. Set `CAMERA_IP` in `config.py` to the camera's address and confirm the Pi can reach it with `ping <CAMERA_IP>`.

---

## Software Requirements

**Python 3.8+** is required.

Install all dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
numpy
scipy
sounddevice
requests
tensorflow          # OR: ai-edge-litert (lighter, preferred on Raspberry Pi)
```

> **Tip for Raspberry Pi:** Use `ai-edge-litert` instead of full TensorFlow to save disk space and memory. The classifier automatically detects and prefers `ai_edge_litert` → `tflite_runtime` → `tensorflow` in that order.

**PortAudio** is required by `sounddevice` (usually pre-installed on Raspberry Pi OS):

```bash
sudo apt install portaudio19-dev   # Debian/Raspberry Pi OS
```

**Optional (for WAV file replay):**

`scipy` is already in `requirements.txt` and provides `resample_poly` for sample rate conversion when replaying WAV files at a non-native sample rate.

---

## Project Structure

```
Sound_Aware_Security_Camera/
│
├── main.py                          # Entry point: audio loop + decision pipeline
├── classifier.py                    # TFLite model wrapper (YAMNet + dense head)
├── doa.py                           # Direction of Arrival via GCC-PHAT
├── camera.py                        # PTZ camera HTTP controller
├── config.py                        # All tunable settings in one place
├── term.py                          # Coloured, timestamped terminal logging
├── doa_check.py                     # Microphone calibration / alignment tool
│
├── models/
│   └── sentry_end_to_end_quantized.tflite   # Quantised TFLite model (~3.7 MB)
│
├── requirements.txt
└── README.md                        # This file
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url> Sound_Aware_Security_Camera
cd Sound_Aware_Security_Camera
```

### 2. (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / Raspberry Pi OS
# or on Windows:
# .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For Raspberry Pi, prefer the lighter TFLite runtime:

```bash
pip install ai-edge-litert          # instead of tensorflow
pip install numpy scipy sounddevice requests
```

### 4. Verify audio devices

```bash
python3 main.py --list-devices
```

Note the index of your I2S microphone input (the two INMP441 mics appear as one 2-channel device) and set `AUDIO_DEVICE` in `config.py`.

### 5. Calibrate the microphones (optional but recommended)

```bash
python3 doa_check.py
```

Follow the on-screen instructions (clap in front, far left, far right). See [doa_check.py](#doa_checkpy--microphone-calibration-tool) for details.

### 6. Configure camera settings

Edit `config.py` and set:
- `CAMERA_IP` — the IP address of the TP-Link camera (the Pi must reach it over the Ethernet link)
- `USERNAME`, `PASSWORD` — camera credentials (default: `admin`/`admin`)
- `MIC_DISTANCE` — actual physical distance in metres between the two microphones

---

## Configuration

All settings live in [`config.py`](config.py). No other file needs editing for a typical deployment.

### Model settings

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/sentry_end_to_end_quantized.tflite` | Path to the TFLite model |
| `MODEL_SAMPLE_RATE` | `16000` | Sample rate the model was trained on (Hz) |
| `MODEL_WINDOW_SECONDS` | `1.0` | Duration of each audio window fed to the model (s) |
| `SUSPICIOUS_THRESHOLD` | `0.60` | Minimum `P(suspicious)` to count as a detection |
| `CONSECUTIVE_HITS` | `1` | Number of consecutive windows above threshold before acting |

### Audio settings

| Parameter | Default | Description |
|---|---|---|
| `AUDIO_DEVICE` | `0` | `sounddevice` input device index |
| `SAMPLE_RATE` | `48000` | Hardware capture sample rate (Hz) |
| `CHANNELS` | `2` | Number of input channels (stereo) |
| `BLOCK_SIZE` | `4800` | Samples per audio callback (= 0.1 s) |
| `HOP_SECONDS` | `0.5` | How often a new window is classified (s) |
| `WARMUP_SECONDS` | `1.0` | Seconds of audio discarded at start (avoids mic click) |
| `RMS_GATE` | `0.05` | Minimum per-channel RMS to pass the loudness gate |

### Direction of Arrival settings

| Parameter | Default | Description |
|---|---|---|
| `MIC_DISTANCE` | `0.25` | Physical distance between microphones (m) |
| `SOUND_SPEED` | `343.0` | Speed of sound (m/s) |
| `DOA_SEGMENT` | `16384` | Samples used for the GCC-PHAT cross-correlation |

### Camera settings

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_IP` | `172.17.137.93` | IP address of the camera |
| `USERNAME` | `admin` | Camera HTTP auth username |
| `PASSWORD` | `admin` | Camera HTTP auth password |
| `HTTP_TIMEOUT` | `1.0` | Timeout for each HTTP request (s) |
| `MOVE_MODE` | `"continuous"` | `"continuous"` (smooth) or `"preset"` (snap to nearest 15°) |
| `PAN_SPEED_RIGHT` | `48.5` | Measured pan speed turning right (deg/s) |
| `PAN_SPEED_LEFT` | `47.5` | Measured pan speed turning left (deg/s) |
| `PAN_START_LAG_RIGHT` | `0.05` | Motor start-up lag, right direction (s) |
| `PAN_START_LAG_LEFT` | `0.03` | Motor start-up lag, left direction (s) |
| `PRESET_SPEED` | `30.0` | Estimated preset-move speed (deg/s) |
| `STOP_RETRIES` | `3` | Number of `move=stop` retries before giving up |
| `HOME_ON_START` | `True` | Go to home preset (0°) when program starts |
| `SNAP_TO_HOME_DEGREES` | `7.5` | Targets within this angle of 0° use the home preset |
| `IDLE_HOME_SECONDS` | `20.0` | Auto-return home after this long with no move (0 = off) |
| `MIN_ANGLE` | `-90.0` | Left pan limit (deg) |
| `MAX_ANGLE` | `90.0` | Right pan limit (deg) |
| `MIN_TURN_DEGREES` | `5.0` | Minimum move size (smaller moves are skipped) |
| `COOLDOWN` | `2.0` | Audio ignored for this long after a move finishes (s) |
| `RETURN_HOME_ON_EXIT` | `True` | Return camera to 0° when the program exits |

### Camera presets map

```python
PRESETS = {
    0: 1, 15: 2, 30: 3, 45: 4, 60: 5, 75: 6, 90: 7,
    -15: 14, -30: 15, -45: 16, -60: 17, -90: 19,
}
HOME_PRESET = 1   # preset 1 = 0 deg (front)
```

Preset 18 (label "285", −75°) is omitted because it stores the same motor position as preset 17 ("300", −60°). Re-save it on the camera's web interface to re-enable it.

---

## Usage

### Live microphone + real camera (normal operation)

```bash
python3 main.py
```

### Dry-run mode (live mic, camera commands printed but not sent)

```bash
python3 main.py --dry-run
```

### Replay a WAV file (camera simulated, no real hardware needed)

```bash
python3 main.py --file path/to/recording.wav
```

The WAV file is automatically resampled to 48 kHz if necessary. Mono files are duplicated to stereo (DOA will always be 0°).

### List audio devices

```bash
python3 main.py --list-devices
```

### Recenter the camera after a crash

If the program crashed while the camera was mid-move, use `--recenter` to tell the system where the camera currently is and return it to 0°:

```bash
python3 main.py --recenter 71.1   # camera is stuck at +71.1 deg; return to 0
```

### Advanced options

```
--threshold FLOAT    P(suspicious) threshold (default: 0.60)
--gate FLOAT         RMS loudness gate (default: 0.05)
--hits INT           Consecutive suspicious windows needed before acting (default: 1)
--no-return          Leave camera at its final position on exit (don't return to 0°)
--device INT         Audio input device index (default: from config.py)
```

### Microphone calibration tool

```bash
python3 doa_check.py
```

Listens for 60 s and prints the lag and estimated angle for every loud sound. No model is used; the camera does not move. Clap 5 times straight ahead (expect ~0°), 5 times far left (expect ~−90°), 5 times far right (expect ~+90°).

---

## Modules In Depth

### `main.py` — Orchestrator & Audio Loop

**Classes:**

- **`Stats`** — Accumulates per-session counters (windows analysed, quiet, skipped, classified, suspicious detections, camera moves, HTTP errors, inference times) and prints a formatted summary at exit.

- **`SoundSentry`** — Owns one each of `SentryClassifier`, `DirectionFinder`, and `CameraController`. Its `process(window, now)` method implements the full decision pipeline described above.

**Functions:**

- **`run_live(sentry, device)`** — Opens a `sounddevice.InputStream` with a ring-buffer callback. Every `HOP_SECONDS` the latest window is copied under a lock and passed to `sentry.process()`. Runs until `KeyboardInterrupt`.

- **`run_file(sentry, path)`** — Reads a WAV file with the standard library `wave` module, resamples with `scipy.signal.resample_poly` if needed, and replays it window-by-window with simulated timestamps.

- **`main()`** — Argument parsing, component construction, startup banner, and top-level exception handling. Always attempts `camera.shutdown()` and prints the session summary on exit, even after errors.

### `classifier.py` — TFLite Inference

**`SentryClassifier`** wraps the quantised TFLite model.

**Runtime detection order:**
1. `ai_edge_litert.interpreter.Interpreter` (Google's lightweight runtime)
2. `tflite_runtime.interpreter.Interpreter`
3. `tensorflow.lite.Interpreter` (full TensorFlow fallback)

**Preprocessing pipeline (`to_model_input`):**
1. Stereo → mono (channel mean)
2. 48 kHz → 16 kHz (3:1 decimation via `resample_poly` with anti-aliasing)
3. Zero-pad or trim to exactly `MODEL_WINDOW_SECONDS × MODEL_SAMPLE_RATE` samples
4. Peak-normalise to [−1, 1]

**`predict(stereo)`** returns `(P(suspicious), inference_time_ms)`.

The interpreter's input tensor is resized at load time from the model's default `[1, 1]` to `[1, 16000]` to match the actual window length.

### `doa.py` — Direction of Arrival (GCC-PHAT)

**`loudest_segment(stereo, length, step=2400)`**  
Slides a window of `length` samples across the stereo signal in steps of 2400 and returns the segment with the highest total energy. This focuses the DOA estimate on the portion of the window that actually caused the detection, rather than averaging over trailing silence.

**`gcc_phat_lag(left, right, max_lag)`**  
Computes the Generalised Cross-Correlation with Phase Transform (GCC-PHAT):

```
GCC-PHAT(τ) = IFFT( FFT(left) · conj(FFT(right)) / |FFT(left) · conj(FFT(right))| )
```

Only lags in `[−max_lag, +max_lag]` are searched, where `max_lag = ceil(d / c × fs) + 1 = 36` samples for 25 cm at 48 kHz. Parabolic interpolation around the peak gives sub-sample lag resolution.

**`DirectionFinder.estimate(stereo)`** returns:

```python
{
    "angle":   float,   # degrees; positive = right
    "lag":     float,   # samples (fractional)
    "dt_ms":   float,   # time delay in ms
    "quality": float,   # peak / mean of |CC| (higher = cleaner signal)
}
```

Angle formula: `angle = degrees(arcsin(clip(dt × c / d, −1, 1)))`

### `camera.py` — PTZ Camera Controller

**`CameraController`** manages all HTTP communication with the camera on a **background worker thread** so the audio stream is never blocked.

**Thread model:**
- The main thread calls `camera.request(plan, now)`, which enqueues a move plan in a `queue.Queue(maxsize=1)`.
- The `_run()` worker thread dequeues plans and calls `_execute()`.
- `camera.ready(now)` returns `True` only when `not moving and now >= busy_until`.
- A `threading.Event` (`_stop`) signals the worker to exit; it also interrupts a move-in-progress during shutdown.

**Two move modes:**

| Mode | Behaviour |
|---|---|
| `"continuous"` | `move=<dir>&move=repeat` → sleep for calculated duration → `move=stop` (retried up to 3×) |
| `"preset"` | `gotoserverpresetno=N` → sleep for estimated travel time + settle |

**Heading tracking:**  
The heading is dead-reckoned (not read back from the camera). It is re-synchronised to absolute 0° at four moments:
- Program start (`HOME_ON_START`)
- Near-0° targets (within `SNAP_TO_HOME_DEGREES`)
- After idle for `IDLE_HOME_SECONDS`
- After a `move=stop` fails (prevents the camera running to its end-stop)

**Shutdown sequence:**
1. Set `_stop` event (interrupts any move in progress; heading is estimated from elapsed time)
2. Join the worker thread (timeout 5 s)
3. If `return_home=True`: execute a blocking preset move to home (absolute re-sync)
4. Otherwise: send `move=stop`

### `config.py` — All Settings

A single flat module of named constants. All other modules import it as `import config as cfg`. No classes, no functions. Edit this file to reconfigure the system.

### `term.py` — Terminal Logging

**`log(message, kind="info", stamp=None)`**  
Thread-safe, colour-coded log line. `stamp` defaults to the current wall-clock time (`HH:MM:SS.d`) but can be overridden (used in file-replay mode to show the file timestamp).

Colour scheme:

| Kind | Colour | Used for |
|---|---|---|
| `"info"` | default | Normal classification results |
| `"ok"` | green | Successful startup, stream open |
| `"warn"` | yellow | Below-threshold events, skipped moves |
| `"alert"` | bold red | Suspicious detections, fatal errors |
| `"camera"` | cyan | Camera move commands and results |
| `"dim"` | dim white | Verbose/debug messages |

**`banner(title, rows)`** — Prints a box with a title and key-value rows. Used for the startup banner and session summary.

Colour output is suppressed automatically if `stdout` is not a TTY (e.g., when logging to a file).

### `doa_check.py` — Microphone Calibration Tool

A standalone script (no model, no camera). It streams audio for `TEST_SECONDS = 60` seconds and, for every 0.5 s window above the RMS gate, prints:

```
[  t s left] rms=0.123  lag=+34.5 samples  q=12.3  angle if 10cm: +87.4  if 25cm: +71.1
```

The `angle if Xcm` columns show what angle that lag would mean for 10 cm and 25 cm mic spacings, making it easy to determine the actual physical spacing from the measured lags at ±90°.

At the end it prints min/max lag observed, with a guide:
- max |lag| ≈ 14 → mics ~10 cm apart  
- max |lag| ≈ 35 → mics ~25 cm apart  
- values beyond expected → echoes or a fixed channel delay

---

## The AI Model

**File:** `models/sentry_end_to_end_quantized.tflite` (~3.7 MB)

**Architecture:**
```
Input: raw_audio [1, 16000] float32  (1 s at 16 kHz, peak-normalised)
   └─► YAMNet (frozen, pre-trained on AudioSet 521 classes)
         produces frame-level 1024-dim embeddings
   └─► Mean pool across frames
   └─► Dense 256, ReLU
   └─► Dropout 0.5 (training only)
   └─► Dense 1, Sigmoid
Output: [1, 1] float32  → P(suspicious)
```

**Training setup:**
- **Base model:** YAMNet (Google, AudioSet, CC-BY 4.0)
- **Dataset:** ESC-50 environmental sound dataset (Piczak 2015)
- **Suspicious classes (label 1):** footsteps, glass breaking, door wood knock, door wood creaks
- **Background classes (label 0):** wind, rain, thunderstorm, engine, car horn, dog, crickets
- **Preprocessing:** highest-energy 1 s chunk per clip; 16 kHz; peak-normalised to [−1, 1]
- **Split:** 80% train / 20% validation, stratified
- **Training environment:** Kaggle notebook (`sound-model-ai-robotics.ipynb`)
- **Export:** end-to-end quantised TFLite (YAMNet preprocessing + head in one file)

**Runtime performance on Raspberry Pi:** ~7–10 ms per 1-second window

**Threshold:** `SUSPICIOUS_THRESHOLD = 0.60` (adjustable via `--threshold`)

---

## Camera Control Details

The TP-Link TL-SC4171G uses an HTTP CGI API at `/cgi-bin/operator/ptzset`. Several engineering challenges were encountered and solved:

| # | Problem | Root Cause | Solution |
|---|---|---|---|
| 1 | Jerky movement in 1.5 s chunks | `move=stop` was needed every 600 ms (PTZ.Delay.PanDelay) | Use `move=<dir>&move=repeat`; motor keeps turning until explicit stop |
| 2 | Camera overshot by ~60% | Assumed speed was 30°/s; actual continuous speed is ~48.5°/s | Calibrated with snapshots at the 15° presets; measured 48.5°/s right, 47.5°/s left |
| 3 | Moves ended too early (short right moves ~1.5° short) | HTTP round-trip time was subtracted from move duration | Time move from before the start request; add `PAN_START_LAG` compensation |
| 4 | Camera spun to its mechanical end-stop | A lost `move=stop` was not retried | Retry stop 3× with 50 ms delay; on total failure, execute absolute home preset re-sync |
| 5 | Heading drift over many moves | Dead-reckoned position accumulates small timing errors | Re-sync to absolute home preset at startup, exit, idle, and near-0° targets |
| 6 | Ctrl+C hung while camera was moving | Move thread was not interrupted | `threading.Event` allows immediate interruption; heading estimated from elapsed time |
| 7 | Camera stuck pointing at a wall | Physical wall at +45°; echoes made DOA read +90° | Moved the entire hardware board to an open area |
| 8 | Duplicate preset motor position | Preset "285" (−75°) stored same position as "300" (−60°) | Omitted from `PRESETS` dict; re-save on camera to fix |

**Preset map** (heading → preset number, from camera's web interface):

| Heading | −90° | −60° | −45° | −30° | −15° | 0° | +15° | +30° | +45° | +60° | +75° | +90° |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Preset # | 19 | 17 | 16 | 15 | 14 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

Camera presets store raw motor positions (~91 encoder units per degree, centre = 16384).

---

## Direction of Arrival Estimation

The system uses two omnidirectional microphones in a linear array. The time delay of arrival (TDOA) between the two channels is converted to an angle using:

```
angle = arcsin( c · Δt / d )
```

Where:
- `c = 343 m/s` — speed of sound
- `d = 0.25 m` — microphone spacing
- `Δt` — time delay (s), positive when the left channel is delayed (sound from the right)

**Maximum possible lag** for 25 cm spacing at 48 kHz:

```
max_lag = ceil(0.25 / 343 × 48000) + 1 = 36 samples
```

Only this range is searched, which improves robustness against noise peaks outside the physically possible range.

**GCC-PHAT** whitens the cross-spectrum before inverse FFT, suppressing frequency-dependent weighting and making the estimator robust to reverberation. Parabolic interpolation around the cross-correlation peak provides sub-sample (fractional) lag resolution.

**Known inaccuracy at the sides:** For a two-microphone array, angular resolution degrades at large angles (near ±90°). Measured results:
- Front (0°): ≈ +3.3° to +3.5° measured ✓
- Far right: ≈ +70° to +78° measured (true: +90°)
- Far left: ≈ −45° to −62° measured (true: −90°)

This is acceptable because the camera's ~53° field of view still covers the sound source at those angles.

---

## Known Limitations

1. **False positives on sharp transients** — The model was trained on ESC-50, which has few "normal but sharp" sounds (claps, door slams, etc.). Any sufficiently loud transient may be classified as suspicious.

2. **Left/right only** — Two microphones provide only a one-dimensional angle (−90° to +90°). Cannot distinguish front from back. Cannot determine elevation.

3. **Reduced accuracy at ±90°** — A two-microphone TDOA estimator has its worst angular resolution at the sides (arcsin linearises near 0° but compresses near ±90°).

4. **Wall echoes** — Nearby reflective surfaces distort the TDOA estimate significantly. Position the hardware away from walls.

5. **Dead-reckoned heading** — The camera position is not read back electronically. If timing is off or a stop command is lost, the heading estimate drifts. Periodic home-preset re-sync mitigates but does not eliminate this.

6. **Single-band RMS gate** — The loudness gate is not frequency-selective. A very quiet suspicious sound could be missed, while a loud but diffuse background (e.g., rain on a metal roof) could fill the gate.

7. **No alert output** — The current system only moves the camera. It does not send notifications, capture snapshots, or record video.

---

## Troubleshooting

**`FileNotFoundError: Model not found: models/sentry_end_to_end_quantized.tflite`**  
Run from the project root directory, or update `MODEL_PATH` in `config.py`.

**Model loads but all sounds score ~0.9 (always suspicious)**  
The audio window reaching the model is very quiet. Check that `RMS_GATE` is set correctly and that `sounddevice` is capturing the right device (`--list-devices`).

**Camera shows `NOT reachable`**  
Check `CAMERA_IP`, `USERNAME`, `PASSWORD` in `config.py`. Check the Ethernet cable between the Pi and camera is seated, and ensure they are on the same network. Try `ping 172.17.137.93` from the Pi.

**Camera moves to the wrong angle**  
Recalibrate `PAN_SPEED_RIGHT` and `PAN_SPEED_LEFT` for your camera's current `PTZ.Speed.PanSpeed` setting. The values in `config.py` were measured at `PanSpeed=100`.

**Microphone channel 0 is on the right physically**  
The sign convention (lag > 0 → left delayed → right turn) assumes channel 0 is the physical left microphone. If your wiring is reversed, negate the lag in `doa.py` or swap the channel indices.

**`sounddevice` throws `PortAudioError`**  
Ensure PortAudio is installed: `sudo apt install portaudio19-dev libportaudio2`. Also try setting `AUDIO_DEVICE` to the correct index from `--list-devices`.

---

## Future Work

- **Custom dataset** — Record suspicious and non-suspicious sounds on the exact same microphone setup; retrain the model for much lower false-positive rates.
- **4-mic array** — A four-microphone array (e.g., ReSpeaker 4-mic) enables 360° direction finding and significantly better angular accuracy.
- **Alert integration** — Send Telegram/email notifications with a snapshot (`/jpg/image.jpg`) on detection.
- **Video recording** — Record an RTSP clip (`rtsp://<ip>:554/video.mp4`) around each detection.
- **Time-of-day arming** — Only act during configured "armed" hours.
- **Web dashboard** — A local web interface showing detections, camera feed, and statistics.
- **Motor encoder feedback** — Query the camera's raw motor position for closed-loop heading tracking.

---

## References

1. **YAMNet** — Howard, A. et al., *"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"*; Google AudioSet; https://tfhub.dev/google/yamnet/1
2. **AudioSet** — Gemmeke, J.F. et al., *"Audio Set: An ontology and human-labeled dataset for audio events"*, ICASSP 2017.
3. **ESC-50** — Piczak, K.J., *"ESC: Dataset for Environmental Sound Classification"*, ACM Multimedia 2015.
4. **GCC-PHAT** — Knapp, C.H. and Carter, G.C., *"The Generalized Correlation Method for Estimation of Time Delay"*, IEEE Trans. ASSP 24(4), 1976.
5. **TensorFlow Lite** — https://www.tensorflow.org/lite
6. **AI Edge LiteRT** — https://ai.google.dev/edge/litert
7. **TP-Link TL-SC4171G** — Product documentation; CGI control via `/cgi-bin/operator/ptzset`.
8. **sounddevice** — https://python-sounddevice.readthedocs.io/

---

*Built for an AI-robotics course project.*
