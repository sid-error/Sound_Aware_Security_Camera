"""
Direction of arrival from two microphones.

Same geometry and sign convention as sound_tracker.py
    lag > 0  ->  left channel is delayed  ->  angle > 0  ->  camera turns right
    angle = arcsin(c * dt / d)

Differences from the reference:
    * GCC-PHAT (FFT based) instead of np.correlate: O(N log N) instead of O(N^2)
      and more robust to room reverberation.
    * Only physically possible lags (+/- d/c seconds) are searched.
    * Parabolic interpolation around the peak gives sub-sample resolution.
    * The loudest DOA_SEGMENT samples of the detection window are used, so the
      angle comes from the sound that triggered the detection.
"""

import numpy as np


def loudest_segment(stereo, length, step=2400):
    if len(stereo) <= length:
        return stereo
    energy = np.square(stereo).sum(axis=1)
    csum = np.concatenate(([0.0], np.cumsum(energy)))
    starts = np.arange(0, len(stereo) - length + 1, step)
    best = starts[np.argmax(csum[starts + length] - csum[starts])]
    return stereo[best:best + length]


def gcc_phat_lag(left, right, max_lag):
    """Lag (in samples, fractional) maximising sum_n left[n+lag] * right[n]."""
    n = len(left) + len(right)
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(left, nfft) * np.conj(np.fft.rfft(right, nfft))
    spec /= np.abs(spec) + 1e-12
    cc = np.fft.irfft(spec, nfft)
    cc = np.concatenate((cc[-max_lag:], cc[:max_lag + 1]))   # lags -max_lag..+max_lag
    k = int(np.argmax(cc))
    shift = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = 0.5 * (y0 - y2) / denom
    peak = cc[k]
    quality = float(peak / (np.mean(np.abs(cc)) + 1e-12))
    return (k - max_lag) + shift, quality


class DirectionFinder:
    def __init__(self, mic_distance, sound_speed, sample_rate, segment):
        self.d = mic_distance
        self.c = sound_speed
        self.fs = sample_rate
        self.segment = segment
        self.max_lag = int(np.ceil(mic_distance / sound_speed * sample_rate)) + 1

    def estimate(self, stereo):
        """Returns dict with angle (deg), lag (samples), dt (s) and peak quality."""
        seg = loudest_segment(stereo, self.segment)
        left = seg[:, 0] - seg[:, 0].mean()
        right = seg[:, 1] - seg[:, 1].mean()
        lag, quality = gcc_phat_lag(left, right, self.max_lag)
        dt = lag / self.fs
        ratio = np.clip(dt * self.c / self.d, -1.0, 1.0)
        return {
            "angle": float(np.degrees(np.arcsin(ratio))),
            "lag": float(lag),
            "dt_ms": dt * 1000,
            "quality": quality,
        }
