"""
PTZ camera control for the TP-Link TL-SC4171G.

Continuous moves (config.MOVE_MODE = "continuous"):
    move=<dir>&move=repeat  ->  wait (angle / speed)  ->  move=stop
"repeat" keeps the motor turning until the stop arrives; without it the camera
stops by itself after PTZ.Delay.PanDelay (600 ms), which is why the old code had
to send 1.5 s chunks. The wait is timed on a monotonic clock from just before the
start request, so HTTP time is not added to the move.

Continuous moves are dead-reckoned, so the heading is re-synced with the home
preset (an absolute motor position) on start, on exit, after a lost stop, for
targets near 0 deg and after the camera has been idle for a while.

The stop is retried, because a lost stop would turn the camera until its end
stop. Shutdown interrupts a move at once and estimates where it stopped.

Live mode runs moves on a worker thread so the audio stream is never blocked.
Dry-run mode prints every command instead of sending it. Simulated mode (file
replay) does not sleep and advances a virtual clock instead.
"""

import queue
import threading
import time

import requests

from term import log


class CameraController:
    def __init__(self, cfg, dry_run=False, simulated=False):
        self.cfg = cfg
        self.dry_run = dry_run or simulated
        self.simulated = simulated
        self.heading = 0.0
        self.moving = False
        self.busy_until = 0.0          # end of move + cooldown (clock seconds)
        self.last_move_end = time.time()
        self.moves = 0
        self.http_errors = 0
        self._session = requests.Session()
        self._session.auth = (cfg.USERNAME, cfg.PASSWORD)
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._worker = None
        if not simulated:
            if cfg.HOME_ON_START:
                self.moving = True
                self._queue.put_nowait(self._home_plan("sync on start"))
            self._worker = threading.Thread(target=self._run, name="camera", daemon=True)
            self._worker.start()

    # ---------------- public API ----------------
    def ready(self, now):
        return not self.moving and now >= self.busy_until

    def plan(self, target_angle):
        """Clamp the target and work out direction/duration. None if too small."""
        target = max(self.cfg.MIN_ANGLE, min(self.cfg.MAX_ANGLE, target_angle))
        preset = None
        if self.cfg.MOVE_MODE == "preset":
            target = min(self.cfg.PRESETS, key=lambda a: abs(a - target))
            preset = self.cfg.PRESETS[target]
        elif abs(target) <= self.cfg.SNAP_TO_HOME_DEGREES:
            target, preset = 0.0, self.cfg.HOME_PRESET
        delta = target - self.heading
        if abs(delta) < self.cfg.MIN_TURN_DEGREES:
            return None
        direction = "right" if delta > 0 else "left"
        if preset is not None:
            duration = abs(delta) / self.cfg.PRESET_SPEED + self.cfg.PRESET_SETTLE
        else:
            speed, lag = ((self.cfg.PAN_SPEED_RIGHT, self.cfg.PAN_START_LAG_RIGHT)
                          if direction == "right"
                          else (self.cfg.PAN_SPEED_LEFT, self.cfg.PAN_START_LAG_LEFT))
            duration = abs(delta) / speed + lag
        return {"target": target, "delta": delta, "direction": direction,
                "duration": duration, "preset": preset}

    def request(self, plan, now, stamp=None):
        """Start a move. Returns False if the camera is busy."""
        with self._lock:
            if not self.ready(now):
                return False
            self.moving = True
        if self.simulated:
            self._execute(plan, stamp)
            self.busy_until = now + plan["duration"] + self.cfg.COOLDOWN
            self.moving = False
            return True
        try:
            self._queue.put_nowait(plan)
        except queue.Full:
            self.moving = False
            return False
        return True

    def check_connection(self):
        """Simple reachability check printed at startup (live mode only)."""
        if self.dry_run:
            return "dry-run (commands printed, not sent)"
        try:
            r = self._session.get(self.cfg.PTZ_URL, params={"move": "stop"},
                                  timeout=self.cfg.HTTP_TIMEOUT)
            return f"reachable (HTTP {r.status_code})"
        except requests.RequestException as e:
            return f"NOT reachable ({type(e).__name__})"

    def return_home(self):
        """Go to the home preset (blocking). It is absolute, so it also works when
        the tracked heading is wrong."""
        log("CAMERA  returning to home position (0 deg)", "camera")
        self.moving = True
        self._execute(self._home_plan("return home"), interruptible=False)
        self.moving = False

    def shutdown(self, return_home=True):
        self._stop.set()               # interrupts a move in progress
        if self._worker:
            self._worker.join(timeout=5)
        if return_home and not self.simulated:
            self.return_home()
        elif not self.dry_run:
            self._stop_motor()

    # ---------------- internals ----------------
    def _home_plan(self, reason):
        delta = -self.heading
        return {"target": 0.0, "delta": delta, "direction": reason,
                "duration": abs(delta) / self.cfg.PRESET_SPEED + self.cfg.PRESET_SETTLE,
                "preset": self.cfg.HOME_PRESET}

    def _run(self):
        while not self._stop.is_set():
            try:
                plan = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._idle_too_long():
                    plan = self._home_plan("idle, re-sync at home")
                else:
                    continue
            self._execute(plan)
            self.last_move_end = time.time()
            self.busy_until = self.last_move_end + self.cfg.COOLDOWN
            self.moving = False
            log(f"CAMERA  ready again after {self.cfg.COOLDOWN:.1f}s cooldown", "dim")

    def _idle_too_long(self):
        """Claim the camera for an idle return home (only if it is away from 0)."""
        if (self.cfg.IDLE_HOME_SECONDS <= 0 or self.heading == 0.0
                or time.time() - self.last_move_end < self.cfg.IDLE_HOME_SECONDS):
            return False
        with self._lock:
            if self.moving:
                return False
            self.moving = True
        return True

    def _send(self, params):
        """Returns (ok, status text)."""
        if self.dry_run:
            return True, "dry-run"
        try:
            r = self._session.get(self.cfg.PTZ_URL, params=params,
                                  timeout=self.cfg.HTTP_TIMEOUT)
            if r.status_code != 200:
                self.http_errors += 1
            return r.status_code == 200, f"HTTP {r.status_code}"
        except requests.RequestException as e:
            self.http_errors += 1
            return False, f"ERROR {type(e).__name__}"

    def _stop_motor(self, stamp=None):
        for attempt in range(1, self.cfg.STOP_RETRIES + 1):
            ok, status = self._send({"move": "stop"})
            log(f"CAMERA    move=stop -> {status}"
                + (f" (attempt {attempt})" if attempt > 1 else ""), "camera", stamp)
            if ok:
                return True
            time.sleep(0.05)
        log("CAMERA  STOP FAILED: camera may keep turning to its end stop", "alert", stamp)
        return False

    def _wait(self, seconds, interruptible):
        """Sleep; returns True if interrupted by shutdown."""
        if self.simulated or seconds <= 0:
            return False
        if interruptible:
            return self._stop.wait(seconds)
        time.sleep(seconds)
        return False

    def _execute(self, plan, stamp=None, interruptible=True):
        if plan["preset"] is not None:
            self._execute_preset(plan, stamp, interruptible)
        else:
            self._execute_continuous(plan, stamp, interruptible)

    def _execute_preset(self, plan, stamp, interruptible):
        log(f"CAMERA  preset {plan['preset']} ({plan['direction']}) "
            f"{self.heading:+.1f} -> {plan['target']:+.1f} deg, ~{plan['duration']:.2f}s",
            "camera", stamp)
        ok, status = self._send({"gotoserverpresetno": plan["preset"]})
        log(f"CAMERA    gotoserverpresetno={plan['preset']} -> {status}", "camera", stamp)
        if not ok:
            return
        # the camera finishes a preset move on its own, so it is not interrupted
        self._wait(plan["duration"], interruptible=False)
        self.heading = plan["target"]
        self.moves += 1
        log(f"CAMERA  at heading {self.heading:+.1f} deg (absolute)", "camera", stamp)

    def _execute_continuous(self, plan, stamp, interruptible):
        direction, duration = plan["direction"], plan["duration"]
        sign = 1.0 if direction == "right" else -1.0
        speed, lag = ((self.cfg.PAN_SPEED_RIGHT, self.cfg.PAN_START_LAG_RIGHT) if sign > 0
                      else (self.cfg.PAN_SPEED_LEFT, self.cfg.PAN_START_LAG_LEFT))
        log(f"CAMERA  turning {direction.upper()} {abs(plan['delta']):.1f} deg "
            f"({self.heading:+.1f} -> {plan['target']:+.1f}) for {duration:.2f}s",
            "camera", stamp)
        t0 = time.monotonic()
        # a list keeps both "move" keys: move=<dir>&move=repeat
        ok, status = self._send([("move", direction), ("move", "repeat")])
        log(f"CAMERA    move={direction}&move=repeat -> {status}", "camera", stamp)
        if not ok:
            self._stop_motor(stamp)     # in case it started anyway
            log("CAMERA  move not started, heading unchanged", "warn", stamp)
            return
        interrupted = self._wait(duration - (time.monotonic() - t0), interruptible)
        elapsed = time.monotonic() - t0
        stopped = self._stop_motor(stamp)
        if interrupted and not self.simulated:
            turned = min(abs(plan["delta"]), max(0.0, elapsed - lag) * speed)
            self.heading += sign * turned
            log(f"CAMERA  move interrupted after {elapsed:.2f}s, "
                f"estimated heading {self.heading:+.1f} deg", "warn", stamp)
        else:
            self.heading = plan["target"]
            log(f"CAMERA  reached heading {self.heading:+.1f} deg", "camera", stamp)
        self.moves += 1
        if not stopped and not self.dry_run:
            # heading is unknown now; the home preset is absolute
            self._execute_preset(self._home_plan("re-sync after failed stop"), stamp, False)
