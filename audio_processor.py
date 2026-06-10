import numpy as np
from scipy import signal
from scipy.signal import find_peaks, savgol_filter
from collections import deque
import threading


GUNSHOT_TRANSIENT_DB = -6  # dB above recent mean → classified as gunshot


# ---------------------------------------------------------------------------
# Frequency Learner
# ---------------------------------------------------------------------------

class FrequencyLearner:
    """
    Accumulates FFT frames from live audio to learn a footstep spectral fingerprint.

    During learning mode the audio callback feeds blocks via update().
    Blocks that are too quiet (silence) or too loud (gunshots) are skipped —
    only moderate-energy sounds (footsteps, light movement) are captured.

    finalize() averages all captured frames, detects spectral peaks, and builds
    a per-bin amplification mask: peaks get boosted, everything else attenuated.
    """

    MIN_RMS = 0.004   # silence floor
    MAX_RMS = 0.70    # gunshot ceiling

    def __init__(self, sample_rate: int = 48000, n_fft: int = 2048):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        self._lock = threading.Lock()
        self.is_learning: bool = False

        self._accumulated = np.zeros(len(self.freqs))
        self._n_frames: int = 0

        # Outputs after finalize()
        self.learned_mask: np.ndarray | None = None
        self.peak_freqs: list[float] = []      # Hz
        self.avg_spectrum: np.ndarray = np.zeros(len(self.freqs))

    # ------------------------------------------------------------------
    # Control

    def start(self) -> None:
        with self._lock:
            self._accumulated[:] = 0.0
            self._n_frames = 0
            self.is_learning = True

    def stop(self) -> None:
        with self._lock:
            self.is_learning = False

    def reset(self) -> None:
        with self._lock:
            self._accumulated[:] = 0.0
            self._n_frames = 0
            self.is_learning = False
            self.learned_mask = None
            self.peak_freqs = []
            self.avg_spectrum = np.zeros(len(self.freqs))

    @property
    def frames_collected(self) -> int:
        with self._lock:
            return self._n_frames

    # ------------------------------------------------------------------
    # Audio-thread update

    def update(self, block: np.ndarray) -> None:
        """Feed one mono audio block from the audio callback thread."""
        if not self.is_learning:
            return
        rms = float(np.sqrt(np.mean(block ** 2)))
        if rms < self.MIN_RMS or rms > self.MAX_RMS:
            return
        mag = np.abs(np.fft.rfft(block, n=self.n_fft))
        with self._lock:
            self._accumulated += mag
            self._n_frames += 1

    # ------------------------------------------------------------------
    # Build mask

    def finalize(self) -> np.ndarray | None:
        """
        Called from GUI thread after learning stops.
        Returns the per-bin mask, or None if too few frames were captured.
        """
        with self._lock:
            if self._n_frames < 10:
                return None
            avg = self._accumulated / self._n_frames

        # Smooth to suppress FFT noise without merging adjacent peaks.
        # Target ~70 Hz bandwidth so peaks 150 Hz apart remain distinct.
        freq_res = self.sample_rate / self.n_fft          # Hz per bin
        win = max(5, int(70.0 / freq_res))
        if win % 2 == 0:
            win += 1                                        # Savgol requires odd
        poly = min(3, win - 1)
        smoothed = savgol_filter(avg, window_length=win, polyorder=poly)
        smoothed = np.maximum(smoothed, 0.0)

        # Minimum distance between peaks: ~100 Hz
        dist = max(3, int(100.0 / freq_res))

        # Detect peaks that are at least 6 % of the maximum
        min_h = smoothed.max() * 0.06
        peaks, _ = find_peaks(smoothed, height=min_h, distance=dist)

        # Build the mask: broad attenuation everywhere, Gaussian bumps at peaks
        mask = np.full(len(self.freqs), 0.12)
        sigma = max(12, len(self.freqs) // 50)  # bandwidth per peak

        for p in peaks:
            # Taller bump for more prominent peaks
            height = 4.0 + 4.0 * (smoothed[p] / (smoothed.max() + 1e-12))
            idx = np.arange(len(self.freqs), dtype=np.float32)
            mask += height * np.exp(-0.5 * ((idx - p) / sigma) ** 2)

        mask = np.clip(mask, 0.05, 14.0)

        self.learned_mask = mask
        self.peak_freqs = [float(self.freqs[p]) for p in peaks[:20]]
        with self._lock:
            self.avg_spectrum = avg.copy()

        return mask

    def get_avg_spectrum(self) -> np.ndarray:
        with self._lock:
            return self._accumulated / max(self._n_frames, 1)


# ---------------------------------------------------------------------------
# Supporting processors
# ---------------------------------------------------------------------------

class SpectralSubtractor:
    """Minimum-statistics background noise estimator."""

    def __init__(self, alpha: float = 0.95, over_sub: float = 2.0):
        self.alpha = alpha
        self.over_sub = over_sub
        self._noise: np.ndarray | None = None

    def update_and_subtract(self, mag: np.ndarray) -> np.ndarray:
        if self._noise is None:
            self._noise = mag.copy()
        else:
            self._noise = np.minimum(
                self.alpha * self._noise + (1 - self.alpha) * mag, mag
            )
        return np.maximum(mag - self.over_sub * self._noise, 0.05 * mag)


class TransientDetector:
    """Flags blocks whose RMS exceeds recent history by threshold_db."""

    def __init__(self, window: int = 20, threshold_db: float = GUNSHOT_TRANSIENT_DB):
        self.threshold_db = threshold_db
        self._history: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def is_transient(self, block: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-12)
        with self._lock:
            self._history.append(rms)
            if len(self._history) < 5:
                return False
            mean_rms = float(np.mean(list(self._history)[:-1]))
        return 20 * np.log10(rms / (mean_rms + 1e-12)) > self.threshold_db


# ---------------------------------------------------------------------------
# Main enhancer
# ---------------------------------------------------------------------------

class FootstepEnhancer:
    """
    Two processing modes:

    FIXED (default):
        Classic 5th-order Butterworth bandpass (150–900 Hz) amplified,
        everything else attenuated.

    LEARNED:
        Per-bin spectral mask derived from FrequencyLearner.  Every FFT bin
        that corresponds to a learned footstep peak is amplified individually;
        all other bins are heavily attenuated.  This adapts to the exact
        footstep signature of the specific game / map / surface.
    """

    FOOTSTEP_LOW_HZ = 150
    FOOTSTEP_HIGH_HZ = 900

    def __init__(self, sample_rate: int = 48000, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Tunable from GUI
        self.footstep_gain: float = 4.0
        self.ambient_suppress: float = 0.25
        self.gunshot_duck: float = 0.10
        self.enabled: bool = True
        self.use_learned: bool = False

        self.learner = FrequencyLearner(
            sample_rate=sample_rate,
            n_fft=max(block_size * 8, 4096),   # higher res → finer peak detection
        )
        self._spectral = SpectralSubtractor()
        self._transient = TransientDetector()

        self._design_filters()
        self._bp_zi: list | None = None
        self._hp_zi: list | None = None
        self._lp_zi: list | None = None

    # ------------------------------------------------------------------
    # Filter design (fixed mode)

    def _design_filters(self) -> None:
        nyq = self.sample_rate / 2.0
        bp_lo = max(self.FOOTSTEP_LOW_HZ / nyq, 1e-4)
        bp_hi = min(self.FOOTSTEP_HIGH_HZ / nyq, 0.9999)
        self._bp_b, self._bp_a = signal.butter(5, [bp_lo, bp_hi], btype="band")

        hp_f = min(self.FOOTSTEP_HIGH_HZ / nyq, 0.9999)
        self._hp_b, self._hp_a = signal.butter(5, hp_f, btype="high")

        lp_f = max(self.FOOTSTEP_LOW_HZ / nyq, 1e-4)
        self._lp_b, self._lp_a = signal.butter(5, lp_f, btype="low")

    def _init_states(self, n_channels: int) -> None:
        self._bp_zi = [signal.lfilter_zi(self._bp_b, self._bp_a).copy() for _ in range(n_channels)]
        self._hp_zi = [signal.lfilter_zi(self._hp_b, self._hp_a).copy() for _ in range(n_channels)]
        self._lp_zi = [signal.lfilter_zi(self._lp_b, self._lp_a).copy() for _ in range(n_channels)]

    def _filter_channel(self, ch: int, audio: np.ndarray):
        bp, self._bp_zi[ch] = signal.lfilter(self._bp_b, self._bp_a, audio, zi=self._bp_zi[ch] * audio[0])
        hp, self._hp_zi[ch] = signal.lfilter(self._hp_b, self._hp_a, audio, zi=self._hp_zi[ch] * audio[0])
        lp, self._lp_zi[ch] = signal.lfilter(self._lp_b, self._lp_a, audio, zi=self._lp_zi[ch] * audio[0])
        return bp, hp, lp

    # ------------------------------------------------------------------
    # Learned-mask spectral processing

    def _apply_mask(self, mono: np.ndarray) -> np.ndarray:
        n = len(mono)
        n_fft = self.learner.n_fft
        spectrum = np.fft.rfft(mono, n=n_fft)
        mag = np.abs(spectrum)
        phase = np.angle(spectrum)

        mask = self.learner.learned_mask
        m = min(len(mask), len(mag))
        mag_out = mag.copy()
        mag_out[:m] *= mask[:m]

        out = np.fft.irfft(mag_out * np.exp(1j * phase), n=n_fft)
        return out[:n].astype(np.float32)

    # ------------------------------------------------------------------
    # Main process

    def process(self, indata: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return indata.astype(np.float32)

        stereo = indata.ndim == 2
        data = indata if stereo else indata[:, np.newaxis]
        n_channels = data.shape[1]

        if self._bp_zi is None or len(self._bp_zi) != n_channels:
            self._init_states(n_channels)

        mono = data.mean(axis=1)
        is_gunshot = self._transient.is_transient(mono)

        # Feed learner regardless of mode
        if self.learner.is_learning:
            self.learner.update(mono)

        out = np.zeros_like(data)

        if self.use_learned and self.learner.learned_mask is not None:
            processed = self._apply_mask(mono)
            if is_gunshot:
                processed *= self.gunshot_duck
            processed = np.tanh(processed)
            for ch in range(n_channels):
                out[:, ch] = processed
        else:
            for ch in range(n_channels):
                bp, hp, lp = self._filter_channel(ch, data[:, ch])
                if is_gunshot:
                    ch_out = (bp * self.footstep_gain + hp + lp) * self.gunshot_duck
                else:
                    ch_out = (
                        bp * self.footstep_gain
                        + hp * self.ambient_suppress
                        + lp * self.ambient_suppress
                    )
                out[:, ch] = np.tanh(ch_out)

        result = out[:, 0] if not stereo else out
        return result.astype(np.float32)
