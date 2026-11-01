"""Timestamped, coloured terminal output used by every module."""

import sys
import threading
import time

_COLOURS = {
    "info": "",
    "ok": "\033[32m",
    "warn": "\033[33m",
    "alert": "\033[1;31m",
    "camera": "\033[36m",
    "dim": "\033[2m",
}
_RESET = "\033[0m"
_USE_COLOUR = sys.stdout.isatty()
_lock = threading.Lock()


def log(message, kind="info", stamp=None):
    """Print one line. `stamp` overrides the wall-clock time (used in file replay)."""
    if stamp is None:
        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}"
    colour = _COLOURS.get(kind, "") if _USE_COLOUR else ""
    reset = _RESET if colour else ""
    with _lock:
        print(f"[{stamp}] {colour}{message}{reset}", flush=True)


def banner(title, rows):
    width = max(len(title), *(len(f"{k}: {v}") for k, v in rows)) + 4
    with _lock:
        print("=" * width)
        print(f"  {title}")
        print("-" * width)
        for key, value in rows:
            print(f"  {key}: {value}")
        print("=" * width, flush=True)
