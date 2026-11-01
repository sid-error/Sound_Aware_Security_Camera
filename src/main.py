#!/usr/bin/env python3
"""
Sound Sentry: the camera turns towards a sound ONLY when the model says it is suspicious.

Pipeline (every HOP_SECONDS, on the last 1 s of stereo audio):
    1. loudness gate (RMS)            quiet -> ignore (the model misfires on quiet audio)
    2. camera busy / cooldown?        yes   -> ignore (avoids hearing its own motor)
    3. model: P(suspicious)           below threshold -> ignore
    4. direction of arrival (TDOA)    angle of the loudest part of the window
    5. camera turns to that angle     on a worker thread

Usage:
    python3 main.py                    live microphone + real camera
    python3 main.py --dry-run          live microphone, camera commands only printed
    python3 main.py --file ../test.wav replay a WAV file (camera simulated)
    python3 main.py --recenter 71.1    camera is at +71.1 deg: turn it back to 0 and exit
    python3 main.py --list-devices     show audio devices
"""

import argparse
import queue
import sys
import threading
import time
import wave

import numpy as np

import config as cfg
from camera import CameraController
from classifier import SentryClassifier
from doa import DirectionFinder
from term import banner, log


class Stats:
    def __init__(self):
        self.windows = self.quiet = self.skipped_busy = 0
        self.classified = self.suspicious = self.moves_requested = 0
        self.small_turns = 0
        self.max_prob = 0.0
        self.infer_ms = []
        self.started = time.time()

    def print_summary(self, camera):
        runtime = time.time() - self.started
        avg = np.mean(self.infer_ms) if self.infer_ms else 0.0
        banner("SESSION SUMMARY", [
            ("Run time", f"{runtime:.1f} s"),
            ("Windows analysed", self.windows),
            ("  quiet (below RMS gate)", self.quiet),
            ("  skipped (camera busy/cooldown)", self.skipped_busy),
            ("  sent to model", self.classified),
            ("Suspicious detections", self.suspicious),
            ("Highest P(suspicious)", f"{self.max_prob:.3f}"),
            (f"Turns too small to move (<{cfg.MIN_TURN_DEGREES:g} deg)", self.small_turns),
            ("Camera moves", camera.moves),
            ("Camera HTTP errors", camera.http_errors),
            ("Final camera heading", f"{camera.heading:+.1f} deg"),
            ("Avg model inference", f"{avg:.1f} ms"),
        ])


class SoundSentry:
    def __init__(self, classifier, finder, camera, threshold, gate, hits_needed):
        self.classifier = classifier
        self.finder = finder
        self.camera = camera
        self.threshold = threshold
        self.gate = gate
        self.hits_needed = hits_needed
        self.hits = 0
        self.stats = Stats()

    def process(self, window, now, stamp=None):
        """Run the full decision pipeline on one stereo window."""
        s = self.stats
        s.windows += 1
        rms_l = float(np.sqrt(np.mean(window[:, 0] ** 2)))
        rms_r = float(np.sqrt(np.mean(window[:, 1] ** 2)))
        level = f"rms L={rms_l:.3f} R={rms_r:.3f}"

        if max(rms_l, rms_r) < self.gate:
            s.quiet += 1
            self.hits = 0
            log(f"{level} | quiet (gate {self.gate}) -> ignored", "dim", stamp)
            return

        if not self.camera.ready(now):
            s.skipped_busy += 1
            log(f"{level} | camera moving/cooldown -> ignored", "dim", stamp)
            return

        prob, ms = self.classifier.predict(window)
        s.classified += 1
        s.infer_ms.append(ms)
        s.max_prob = max(s.max_prob, prob)

        if prob < self.threshold:
            self.hits = 0
            log(f"{level} | P(suspicious)={prob:.3f} ({ms:.1f} ms) -> normal sound, camera stays",
                "info", stamp)
            return

        self.hits += 1
        if self.hits < self.hits_needed:
            log(f"{level} | P(suspicious)={prob:.3f} -> hit {self.hits}/{self.hits_needed}, waiting",
                "warn", stamp)
            return
        self.hits = 0
        s.suspicious += 1

        doa = self.finder.estimate(window)
        log(f"{level} | P(suspicious)={prob:.3f} ({ms:.1f} ms) -> SUSPICIOUS SOUND", "alert", stamp)
        log(f"  DOA: lag={doa['lag']:+.2f} samples  dt={doa['dt_ms']:+.3f} ms  "
            f"angle={doa['angle']:+.1f} deg  peak quality={doa['quality']:.1f}  "
            f"| camera at {self.camera.heading:+.1f} deg", "alert", stamp)

        plan = self.camera.plan(doa["angle"])
        if plan is None:
            s.small_turns += 1
            log(f"  camera already pointing there (<{cfg.MIN_TURN_DEGREES} deg) -> no move",
                "warn", stamp)
            return
        if self.camera.request(plan, now, stamp):
            s.moves_requested += 1


# ---------------- live microphone ----------------
def run_live(sentry, device):
    window_len = int(cfg.SAMPLE_RATE * cfg.MODEL_WINDOW_SECONDS)
    buffer = np.zeros((window_len, cfg.CHANNELS), dtype=np.float32)
    lock = threading.Lock()
    status_q = queue.Queue()
    import sounddevice as sd

    def callback(indata, frames, time_info, status):
        nonlocal buffer
        if status:
            status_q.put(str(status))
        with lock:
            buffer = np.roll(buffer, -frames, axis=0)
            buffer[-frames:] = indata

    stream = sd.InputStream(device=device, samplerate=cfg.SAMPLE_RATE, channels=cfg.CHANNELS,
                            blocksize=cfg.BLOCK_SIZE, dtype="float32", callback=callback)
    with stream:
        log(f"Microphone stream open: {sd.query_devices(device)['name']} "
            f"@ {cfg.SAMPLE_RATE} Hz, {cfg.CHANNELS} ch", "ok")
        log(f"Warming up: discarding first {cfg.WARMUP_SECONDS:.1f} s (mic start-up click), "
            f"then filling {cfg.MODEL_WINDOW_SECONDS:.1f} s buffer...", "dim")
        time.sleep(cfg.WARMUP_SECONDS + cfg.MODEL_WINDOW_SECONDS)
        log("Listening. Press Ctrl+C to stop.", "ok")
        next_tick = time.time()
        while True:
            next_tick += cfg.HOP_SECONDS
            while not status_q.empty():
                log(f"Audio status: {status_q.get()}", "warn")
            with lock:
                window = buffer.copy()
            sentry.process(window, time.time())
            time.sleep(max(0.0, next_tick - time.time()))


# ---------------- file replay ----------------
def read_wav(path):
    with wave.open(path, "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dtype = {2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width {width * 8} bits")
    data = np.frombuffer(raw, dtype=dtype).reshape(-1, ch).astype(np.float32)
    data /= float(np.iinfo(dtype).max)
    if ch == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2], rate


def run_file(sentry, path):
    from scipy.signal import resample_poly
    audio, rate = read_wav(path)
    log(f"Loaded {path}: {len(audio) / rate:.2f} s, {rate} Hz", "ok")
    if rate != cfg.SAMPLE_RATE:
        audio = resample_poly(audio, cfg.SAMPLE_RATE, rate, axis=0).astype(np.float32)
        log(f"Resampled {rate} -> {cfg.SAMPLE_RATE} Hz", "dim")
    window_len = int(cfg.SAMPLE_RATE * cfg.MODEL_WINDOW_SECONDS)
    hop = int(cfg.SAMPLE_RATE * cfg.HOP_SECONDS)
    if len(audio) < window_len:
        audio = np.pad(audio, ((0, window_len - len(audio)), (0, 0)))
    for end in range(window_len, len(audio) + 1, hop):
        t = end / cfg.SAMPLE_RATE
        sentry.process(audio[end - window_len:end], t, stamp=f"file {t:6.2f}s")


def main():
    ap = argparse.ArgumentParser(description="Move the camera only for suspicious sounds")
    ap.add_argument("--dry-run", action="store_true", help="print camera commands, don't send them")
    ap.add_argument("--file", help="replay a WAV file instead of the microphone")
    ap.add_argument("--device", type=int, default=cfg.AUDIO_DEVICE, help="audio input device index")
    ap.add_argument("--threshold", type=float, default=cfg.SUSPICIOUS_THRESHOLD)
    ap.add_argument("--gate", type=float, default=cfg.RMS_GATE, help="RMS loudness gate")
    ap.add_argument("--hits", type=int, default=cfg.CONSECUTIVE_HITS,
                    help="consecutive suspicious windows needed before moving")
    ap.add_argument("--no-return", action="store_true",
                    help="leave the camera where it is on exit (default: return to 0 deg)")
    ap.add_argument("--recenter", type=float, metavar="DEG",
                    help="camera is currently at DEG (e.g. after a crash): turn it back to 0 and exit")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    return_home = cfg.RETURN_HOME_ON_EXIT and not args.no_return

    if args.recenter is not None:
        camera = CameraController(cfg, dry_run=args.dry_run)
        camera.heading = args.recenter
        log(f"Recentering: camera assumed at {args.recenter:+.1f} deg "
            f"({camera.check_connection()})", "camera")
        camera.shutdown(return_home=True)
        return

    log("Loading model...", "dim")
    classifier = SentryClassifier(cfg.MODEL_PATH, cfg.MODEL_SAMPLE_RATE,
                                  cfg.MODEL_WINDOW_SECONDS, cfg.SAMPLE_RATE)
    finder = DirectionFinder(cfg.MIC_DISTANCE, cfg.SOUND_SPEED, cfg.SAMPLE_RATE, cfg.DOA_SEGMENT)
    camera = CameraController(cfg, dry_run=args.dry_run, simulated=bool(args.file))
    sentry = SoundSentry(classifier, finder, camera, args.threshold, args.gate, args.hits)

    mode = (f"file replay ({args.file})" if args.file
            else "live mic, DRY-RUN camera" if args.dry_run else "live mic + camera")
    banner("SOUND SENTRY", [
        ("Mode", mode),
        *classifier.describe(),
        ("Decision", f"RMS >= {args.gate} AND P(suspicious) >= {args.threshold} "
                     f"for {args.hits} window(s)"),
        ("Window / hop", f"{cfg.MODEL_WINDOW_SECONDS} s / {cfg.HOP_SECONDS} s"),
        ("Microphones", f"device {args.device}, {cfg.MIC_DISTANCE} m apart, "
                        f"max lag +/-{finder.max_lag} samples"),
        ("Camera", f"{cfg.PTZ_URL} -> {camera.check_connection()}"),
        ("On exit", "return camera to 0 deg" if return_home else "leave camera where it is"),
        ("Camera limits", f"{cfg.MIN_ANGLE:+.0f}..{cfg.MAX_ANGLE:+.0f} deg, "
                          f"{cfg.MOVE_MODE} moves, "
                          f"{cfg.PAN_SPEED_LEFT}/{cfg.PAN_SPEED_RIGHT} deg/s L/R, "
                          f"cooldown {cfg.COOLDOWN} s"),
    ])

    try:
        if args.file:
            run_file(sentry, args.file)
        else:
            run_live(sentry, args.device)
    except KeyboardInterrupt:
        print()
        log("Stopping (Ctrl+C)...", "warn")
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}", "alert")
        camera.shutdown(return_home)
        sentry.stats.print_summary(camera)
        sys.exit(1)
    camera.shutdown(return_home)
    sentry.stats.print_summary(camera)


if __name__ == "__main__":
    main()
