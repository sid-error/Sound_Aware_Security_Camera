"""
All tunable settings for Sound Sentry in one place.

Camera, microphone geometry and motor calibration values are taken from
sound_tracker.py (the reference implementation) so behaviour stays the same.
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Model ----------------
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "sentry_end_to_end_quantized.tflite")
MODEL_SAMPLE_RATE = 16000        # the notebook trained on 16 kHz audio
MODEL_WINDOW_SECONDS = 1.0       # the notebook trained on 1-second chunks
SUSPICIOUS_THRESHOLD = 0.60      # P(suspicious) needed to count as a detection
CONSECUTIVE_HITS = 1             # windows in a row above threshold before acting

# ---------------- Audio ----------------
AUDIO_DEVICE = 0                 # Google voiceHAT / ReSpeaker card (hw:2,0)
SAMPLE_RATE = 48000              # hardware native rate (same as sound_tracker.py)
CHANNELS = 2                     # column 0 = left mic, column 1 = right mic
BLOCK_SIZE = 4800                # 0.1 s per callback
HOP_SECONDS = 0.5                # classify a new 1 s window every 0.5 s
# Opening the mic stream produces a click the model scores as suspicious
# (seen in the first live test), so the first audio is discarded.
WARMUP_SECONDS = 1.0

# Loudness gate. Required: the model scores silence/quiet input as "suspicious",
# so quiet windows must never reach it. Same value as sound_tracker.py.
RMS_GATE = 0.05

# ---------------- Direction of arrival ----------------
MIC_DISTANCE = 0.25              # metres between the two microphones
SOUND_SPEED = 343.0              # m/s
DOA_SEGMENT = 16384              # samples used for TDOA (same block as sound_tracker.py)

# ---------------- Camera ----------------
CAMERA_IP = "172.17.137.93"
USERNAME = "admin"
PASSWORD = "admin"
PTZ_URL = f"http://{CAMERA_IP}/cgi-bin/operator/ptzset"
HTTP_TIMEOUT = 1.0

# "continuous": move=<dir>&move=repeat, wait angle/speed, move=stop. One smooth
#               move to any angle. ("repeat" is required: without it the camera
#               stops by itself after PTZ.Delay.PanDelay = 600 ms.)
# "preset":     one gotoserverpresetno command to the nearest 15 deg preset.
MOVE_MODE = "continuous"

# Continuous pan speed, measured 2026-10-12 against the presets with
# PTZ.Speed.PanSpeed=100 (moves of 0.5-4 s, fit error ~1 deg). Re-measure if the
# camera's PanSpeed setting is changed. Preset moves are slower (~30 deg/s).
PAN_SPEED_RIGHT = 48.5           # deg/s
PAN_SPEED_LEFT = 47.5            # deg/s
# s between sending the start and the motor turning. The right lag is longer:
# measured 2026-10-12, short right moves (0.25-0.9 s) fell ~1.5 deg short at 0.03.
PAN_START_LAG_RIGHT = 0.05
PAN_START_LAG_LEFT = 0.03
PRESET_SPEED = 30.0              # deg/s, used to estimate preset travel time
STOP_RETRIES = 3                 # a lost "stop" would run the camera to its end stop

# Continuous moves are dead-reckoned, so small timing errors add up. The home
# preset is an absolute position, so it is used to re-sync the heading:
HOME_ON_START = True             # go to the home preset when the program starts
SNAP_TO_HOME_DEGREES = 7.5       # targets this close to 0 go to the home preset
IDLE_HOME_SECONDS = 20.0        # return home after this long with no move (0 = off)

# Heading (deg, + = right) -> preset number, from the presets saved on the camera
# (labels 0..90 = right, 345..270 = left). Preset 18 (label "285", -75 deg) is
# left out: it holds the same motor position as 300. Re-save it on the camera's
# web page, then add  -75: 18  back here.
PRESETS = {
    0: 1, 15: 2, 30: 3, 45: 4, 60: 5, 75: 6, 90: 7,
    -15: 14, -30: 15, -45: 16, -60: 17, -90: 19,
}
HOME_PRESET = 1                  # preset for 0 deg
PRESET_SETTLE = 0.5              # extra wait after the estimated preset travel time

MIN_ANGLE = -90.0
MAX_ANGLE = 90.0
MIN_TURN_DEGREES = 5.0           # smaller moves last less than the HTTP round trip
COOLDOWN = 2.0                   # seconds to ignore audio after a move finishes
RETURN_HOME_ON_EXIT = True       # turn back to 0 deg when the program stops
