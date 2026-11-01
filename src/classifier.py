"""
Wraps sentry_end_to_end_quantized.tflite (built by sound-model-ai-robotics.ipynb).

The model is YAMNet + a dense head in one file:
    input  raw_audio  [1, N] float32, 16 kHz mono
    output            [1, 1] float32, P(suspicious) from a sigmoid

Preprocessing must match the notebook exactly: a 1-second window at 16 kHz,
peak-normalised to [-1, 1]. Without normalisation (or on silence) the model
returns high scores for quiet audio, so callers must apply the RMS gate first.
"""

import os
import time
import warnings

import numpy as np
from scipy.signal import resample_poly

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _load_interpreter_class():
    for module, attr in (("ai_edge_litert.interpreter", "Interpreter"),
                         ("tflite_runtime.interpreter", "Interpreter")):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr), module.split(".")[0]
        except ImportError:
            pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tensorflow as tf
    return tf.lite.Interpreter, f"tensorflow {tf.__version__}"


class SentryClassifier:
    def __init__(self, model_path, model_rate, window_seconds, input_rate):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if input_rate % model_rate:
            raise ValueError(f"Input rate {input_rate} must be a multiple of {model_rate}")

        interpreter_cls, self.runtime = _load_interpreter_class()
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.interpreter = interpreter_cls(model_path=model_path)
        self.window = int(model_rate * window_seconds)
        self.decimation = input_rate // model_rate

        self._in = self.interpreter.get_input_details()[0]
        self._out = self.interpreter.get_output_details()[0]
        # The model's default input shape is [1, 1]; resize it to the real window.
        self.interpreter.resize_tensor_input(self._in["index"], [1, self.window])
        self.interpreter.allocate_tensors()
        self.load_ms = (time.perf_counter() - start) * 1000
        self.model_path = model_path

    def describe(self):
        size_mb = os.path.getsize(self.model_path) / 1024 / 1024
        return [
            ("Model", f"{os.path.basename(self.model_path)} ({size_mb:.2f} MB)"),
            ("Runtime", self.runtime),
            ("Model input", f"{self._in['name']} [1, {self.window}] {self._in['dtype'].__name__}"),
            ("Model output", f"{self._out['name']} {[int(d) for d in self._out['shape']]} = P(suspicious)"),
            ("Load time", f"{self.load_ms:.0f} ms"),
        ]

    def to_model_input(self, stereo):
        """48 kHz stereo -> 16 kHz mono, anti-aliased, fixed length, peak-normalised."""
        mono = stereo.mean(axis=1) if stereo.ndim == 2 else stereo
        mono = resample_poly(mono, 1, self.decimation).astype(np.float32)
        if len(mono) < self.window:
            mono = np.pad(mono, (0, self.window - len(mono)))
        mono = mono[-self.window:]
        peak = np.max(np.abs(mono))
        if peak > 0:
            mono = mono / peak
        return mono

    def predict(self, stereo):
        """Returns (P(suspicious), inference time in ms)."""
        x = self.to_model_input(stereo)
        start = time.perf_counter()
        self.interpreter.set_tensor(self._in["index"], x[np.newaxis, :])
        self.interpreter.invoke()
        prob = float(self.interpreter.get_tensor(self._out["index"]).reshape(-1)[0])
        return prob, (time.perf_counter() - start) * 1000
