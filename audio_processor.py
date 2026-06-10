import numpy as np
from scipy import signal
from collections import deque
import threading


# Frequency ranges (Hz) identified from CoD audio analysis
FOOTSTEP_LOW_HZ = 150
FOOTSTEP_HIGH_HZ = 900
GUNSHOT_TRANSIENT_DB = -6  # dB above mean → classified as gunshot/explosion


class SpectralSubtractor:
    """Estimates and removes stationary background noise (music, ambience)."""

    def __init__(self, sample_rate: int, n_fft: int = 1024, over_subtraction: float = 2.0):
        self.n_fft = n_fft
        self.over_subtraction = over_subtraction
        self.noise_estimate = None
        self.alpha = 0.95  # smoothing factor for noise estimation

    def update_noise(self, magnitude_spectrum: np.ndarray) -> None:
        if self.noise_estimate is None:
            self.noise_estimate = magnitude_spectrum.copy()
        else:
            # Minimum statistics: slowly track quietest spectrum seen
            self.noise_estimate = np.minimum(
                self.alpha * self.noise_estimate + (1 - self.alpha) * magnitude_spectrum,
                magnitude_spectrum,
            )

    def subtract(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        if self.noise_estimate is None:
            return magnitude_spectrum
        cleaned = magnitude_spectrum - self.over_subtraction * self.noise_estimate
        # Half-wave rectify (no negative values)
        return np.maximum(cleaned, 0.05 * magnitude_spectrum)


class TransientDetector:
    """Detects sudden loud transients (gunshots, explosions) via energy ratio."""

    def __init__(self, window: int = 20):
        self.energy_history = deque(maxlen=window)
        self.lock = threading.Lock()

    def is_transient(self, block: np.ndarray, threshold_db: float = GUNSHOT_TRANSIENT_DB) -> bool:
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-12)
        with self.lock:
            self.energy_history.append(rms)
            if len(self.energy_history) < 5:
                return False
            mean_rms = float(np.mean(list(self.energy_history)[:-1]))
        ratio_db = 20 * np.log10(rms / (mean_rms + 1e-12))
        return ratio_db > threshold_db


class FootstepEnhancer:
    """
    Real-time audio processor that:
      1. Amplifies footstep frequencies (150–900 Hz).
      2. Attenuates background ambience via spectral subtraction.
      3. Ducks gunshots / explosions using transient detection.
    """

    def __init__(self, sample_rate: int = 48000, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size

        # User-adjustable parameters (changed from GUI thread — no locking needed
        # since reads/writes of float are atomic in CPython)
        self.footstep_gain = 4.0      # multiplier for footstep band
        self.ambient_suppress = 0.25  # gain for non-footstep content
        self.gunshot_duck = 0.10      # gain during detected transient
        self.enabled = True

        # Internal state
        self._spectral = SpectralSubtractor(sample_rate)
        self._transient = TransientDetector()
        self._design_filters()

        # Per-channel IIR filter states (initialised lazily on first block)
        self._bp_zi: list[np.ndarray] | None = None
        self._hp_zi: list[np.ndarray] | None = None
        self._lp_zi: list[np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Filter design
    # ------------------------------------------------------------------

    def _design_filters(self) -> None:
        nyq = self.sample_rate / 2.0

        bp_lo = max(FOOTSTEP_LOW_HZ / nyq, 1e-4)
        bp_hi = min(FOOTSTEP_HIGH_HZ / nyq, 0.9999)
        self._bp_b, self._bp_a = signal.butter(5, [bp_lo, bp_hi], btype="band")

        hp_freq = min(FOOTSTEP_HIGH_HZ / nyq, 0.9999)
        self._hp_b, self._hp_a = signal.butter(5, hp_freq, btype="high")

        lp_freq = max(FOOTSTEP_LOW_HZ / nyq, 1e-4)
        self._lp_b, self._lp_a = signal.butter(5, lp_freq, btype="low")

    def _init_states(self, n_channels: int) -> None:
        zi_bp = signal.lfilter_zi(self._bp_b, self._bp_a)
        zi_hp = signal.lfilter_zi(self._hp_b, self._hp_a)
        zi_lp = signal.lfilter_zi(self._lp_b, self._lp_a)
        self._bp_zi = [zi_bp.copy() for _ in range(n_channels)]
        self._hp_zi = [zi_hp.copy() for _ in range(n_channels)]
        self._lp_zi = [zi_lp.copy() for _ in range(n_channels)]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _process_channel(self, ch_idx: int, audio: np.ndarray) -> np.ndarray:
        """Filter a single channel; updates persistent IIR states."""
        bp_out, self._bp_zi[ch_idx] = signal.lfilter(
            self._bp_b, self._bp_a, audio, zi=self._bp_zi[ch_idx] * audio[0]
        )
        hp_out, self._hp_zi[ch_idx] = signal.lfilter(
            self._hp_b, self._hp_a, audio, zi=self._hp_zi[ch_idx] * audio[0]
        )
        lp_out, self._lp_zi[ch_idx] = signal.lfilter(
            self._lp_b, self._lp_a, audio, zi=self._lp_zi[ch_idx] * audio[0]
        )
        return bp_out, hp_out, lp_out

    def process(self, indata: np.ndarray) -> np.ndarray:
        """
        Process one block of audio.

        Args:
            indata: float32 array, shape (frames,) or (frames, channels)

        Returns:
            Processed float32 array with the same shape.
        """
        if not self.enabled:
            return indata.astype(np.float32)

        stereo = indata.ndim == 2
        if stereo:
            n_channels = indata.shape[1]
            frames = indata.shape[0]
        else:
            n_channels = 1
            frames = indata.shape[0]
            indata = indata[:, np.newaxis]

        if self._bp_zi is None or len(self._bp_zi) != n_channels:
            self._init_states(n_channels)

        # Detect transient on mixed-down signal
        mono = indata.mean(axis=1)
        is_gunshot = self._transient.is_transient(mono)

        # Spectral noise estimation (on mono)
        n_fft = 1024
        mag = np.abs(np.fft.rfft(mono, n=n_fft))
        self._spectral.update_noise(mag)

        out = np.zeros_like(indata)

        for ch in range(n_channels):
            ch_audio = indata[:, ch]
            bp, hp, lp = self._process_channel(ch, ch_audio)

            if is_gunshot:
                # Heavy attenuation during gunshot; preserve a whisper of footstep
                channel_out = (
                    bp * self.footstep_gain * self.gunshot_duck
                    + (hp + lp) * self.gunshot_duck
                )
            else:
                channel_out = (
                    bp * self.footstep_gain
                    + hp * self.ambient_suppress
                    + lp * self.ambient_suppress
                )

            # Soft limiter — avoids hard clipping
            out[:, ch] = np.tanh(channel_out)

        return (out[:, 0] if not stereo else out).astype(np.float32)
