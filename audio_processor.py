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

    def __init__(self, sample_rate: int = 48000, n_fft: int = 4096):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        self._lock = threading.Lock()
        self.is_learning: bool = False

        self._accumulated = np.zeros(len(self.freqs))
        self._n_frames: int = 0

        # Outputs after finalize()
        self.learned_mask: np.ndarray | None = None
        self.mask_version: int = 0          # bumped on every finalize/reset
        self.peak_freqs: list[float] = []   # Hz
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
            self.mask_version += 1
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

        # Build the mask: moderate attenuation everywhere, Gaussian bumps at
        # peaks.  Gains are kept modest (≤ 5×) — extreme per-bin gain ratios
        # produce audible musical-noise artifacts.
        mask = np.full(len(self.freqs), 0.20)
        sigma = max(12, len(self.freqs) // 50)  # bandwidth per peak

        for p in peaks:
            height = 2.0 + 2.5 * (smoothed[p] / (smoothed.max() + 1e-12))
            idx = np.arange(len(self.freqs), dtype=np.float32)
            mask += height * np.exp(-0.5 * ((idx - p) / sigma) ** 2)

        mask = np.clip(mask, 0.1, 5.0)

        self.learned_mask = mask
        self.mask_version += 1
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

class TransientDetector:
    """
    Flags blocks whose RMS exceeds recent history by threshold_db.
    Once triggered, stays "hot" for hold_blocks so the whole gunshot tail
    is ducked, not just its first block.
    """

    def __init__(self, window: int = 20, threshold_db: float = GUNSHOT_TRANSIENT_DB,
                 hold_blocks: int = 12):
        self.threshold_db = threshold_db
        self.hold_blocks = hold_blocks
        self._history: deque[float] = deque(maxlen=window)
        self._hold = 0
        self._lock = threading.Lock()

    def is_transient(self, block: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-12)
        with self._lock:
            self._history.append(rms)
            if len(self._history) < 5:
                return False
            mean_rms = float(np.mean(list(self._history)[:-1]))
            if 20 * np.log10(rms / (mean_rms + 1e-12)) > self.threshold_db:
                self._hold = self.hold_blocks
            if self._hold > 0:
                self._hold -= 1
                return True
            return False


class GainSmoother:
    """
    Exponentially smoothed gain envelope: fast attack (duck quickly when a
    gunshot hits), slow release (fade back in) — eliminates pumping clicks
    caused by per-block binary gain switching.
    """

    def __init__(self, sample_rate: int, attack_ms: float = 4.0,
                 release_ms: float = 220.0):
        self._attack = float(np.exp(-1.0 / (sample_rate * attack_ms / 1000.0)))
        self._release = float(np.exp(-1.0 / (sample_rate * release_ms / 1000.0)))
        self._current = 1.0

    def ramp(self, frames: int, target: float) -> np.ndarray:
        """Return per-sample gain ramp moving toward target."""
        coef = self._attack if target < self._current else self._release
        n = np.arange(1, frames + 1, dtype=np.float64)
        gains = target + (self._current - target) * coef ** n
        self._current = float(gains[-1])
        return gains


class SoftLimiter:
    """
    Look-at-block peak limiter with smoothed gain — replaces the old tanh
    waveshaper which distorted everything above ~0.5.
    """

    def __init__(self, sample_rate: int, threshold: float = 0.90):
        self.threshold = threshold
        self._smoother = GainSmoother(sample_rate, attack_ms=1.0, release_ms=120.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) + 1e-12
        target = 1.0 if peak <= self.threshold else self.threshold / peak
        gains = self._smoother.ramp(x.shape[0], target)
        if x.ndim == 2:
            gains = gains[:, np.newaxis]
        out = x * gains
        # Final safety clip — should rarely engage
        return np.clip(out, -1.0, 1.0)


class OLAMaskFilter:
    """
    Overlap-add STFT filter for one audio channel.

    The old implementation did a raw FFT→mask→IFFT on each 512-sample block
    with no window and no overlap, which created loud clicks/buzzing at every
    block boundary.  This version uses a periodic Hann window with 50 %
    overlap (COLA-compliant), so reconstruction is artifact-free.

    Adds hop-size samples of latency (~10.7 ms at 48 kHz / 512).
    """

    def __init__(self, hop: int):
        self.hop = hop
        self.win_len = hop * 2
        self.window = signal.windows.hann(self.win_len, sym=False)
        self.freqs = None  # set by set_mask via sample_rate
        self._in_buf = np.zeros(self.win_len, dtype=np.float64)
        self._out_buf = np.zeros(self.win_len, dtype=np.float64)
        self._mask: np.ndarray | None = None

    def set_mask(self, mask_freqs: np.ndarray, mask: np.ndarray,
                 sample_rate: int) -> None:
        """Resample the learner's mask onto this filter's FFT bins."""
        own_freqs = np.fft.rfftfreq(self.win_len, 1.0 / sample_rate)
        self._mask = np.interp(own_freqs, mask_freqs, mask)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Process one block of up to hop samples; returns len(x) samples."""
        n = len(x)
        if n < self.hop:
            # Partial block (e.g. end of a file) — zero-pad, then trim output
            padded = np.zeros(self.hop, dtype=np.float64)
            padded[:n] = x
            return self.process(padded)[:n]

        # Slide input buffer
        self._in_buf[:-self.hop] = self._in_buf[self.hop:]
        self._in_buf[-self.hop:] = x

        frame = self._in_buf * self.window
        spec = np.fft.rfft(frame)
        if self._mask is not None:
            spec *= self._mask
        y = np.fft.irfft(spec, n=self.win_len)

        # Slide output accumulator and add new frame
        self._out_buf[:-self.hop] = self._out_buf[self.hop:]
        self._out_buf[-self.hop:] = 0.0
        self._out_buf += y

        return self._out_buf[:self.hop].copy()


# ---------------------------------------------------------------------------
# Main enhancer
# ---------------------------------------------------------------------------

class FootstepEnhancer:
    """
    Two processing modes:

    FIXED (default):
        5th-order Butterworth bandpass (150–900 Hz) amplified,
        everything else attenuated.

    LEARNED:
        Per-bin spectral mask derived from FrequencyLearner, applied through
        a windowed overlap-add STFT (artifact-free reconstruction).

    Both modes share a smoothed gunshot-duck envelope and a soft peak limiter
    instead of hard tanh waveshaping.
    """

    FOOTSTEP_LOW_HZ = 150
    FOOTSTEP_HIGH_HZ = 900

    def __init__(self, sample_rate: int = 48000, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Tunable from GUI
        self.footstep_gain: float = 3.0
        self.ambient_suppress: float = 0.25
        self.gunshot_duck: float = 0.10
        self.enabled: bool = True
        self.use_learned: bool = False

        self.learner = FrequencyLearner(
            sample_rate=sample_rate,
            n_fft=max(block_size * 8, 4096),
        )

        self._transient = TransientDetector()
        self._duck = GainSmoother(sample_rate, attack_ms=4.0, release_ms=220.0)
        self._limiter = SoftLimiter(sample_rate)

        self._design_filters()
        self._bp_zi: list | None = None
        self._hp_zi: list | None = None
        self._lp_zi: list | None = None

        self._ola: list[OLAMaskFilter] | None = None
        self._ola_mask_version: int = -1

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
        # Zero initial state — filters settle within a few ms.  The state
        # arrays returned by lfilter are carried forward untouched between
        # blocks (re-scaling them each block corrupts the filter and clicks).
        self._bp_zi = [np.zeros(max(len(self._bp_a), len(self._bp_b)) - 1)
                       for _ in range(n_channels)]
        self._hp_zi = [np.zeros(max(len(self._hp_a), len(self._hp_b)) - 1)
                       for _ in range(n_channels)]
        self._lp_zi = [np.zeros(max(len(self._lp_a), len(self._lp_b)) - 1)
                       for _ in range(n_channels)]

    def _filter_channel(self, ch: int, audio: np.ndarray):
        bp, self._bp_zi[ch] = signal.lfilter(self._bp_b, self._bp_a, audio,
                                             zi=self._bp_zi[ch])
        hp, self._hp_zi[ch] = signal.lfilter(self._hp_b, self._hp_a, audio,
                                             zi=self._hp_zi[ch])
        lp, self._lp_zi[ch] = signal.lfilter(self._lp_b, self._lp_a, audio,
                                             zi=self._lp_zi[ch])
        return bp, hp, lp

    # ------------------------------------------------------------------
    # Learned-mask OLA setup

    def _ensure_ola(self, n_channels: int, hop: int) -> None:
        rebuild = (
            self._ola is None
            or len(self._ola) != n_channels
            or self._ola[0].hop != hop
        )
        if rebuild:
            self._ola = [OLAMaskFilter(hop) for _ in range(n_channels)]
            self._ola_mask_version = -1

        if self._ola_mask_version != self.learner.mask_version:
            mask = self.learner.learned_mask
            if mask is not None:
                for f in self._ola:
                    f.set_mask(self.learner.freqs, mask, self.sample_rate)
            self._ola_mask_version = self.learner.mask_version

    # ------------------------------------------------------------------
    # Main process

    def process(self, indata: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return indata.astype(np.float32)

        stereo = indata.ndim == 2
        data = indata if stereo else indata[:, np.newaxis]
        frames, n_channels = data.shape

        if self._bp_zi is None or len(self._bp_zi) != n_channels:
            self._init_states(n_channels)

        mono = data.mean(axis=1)
        is_gunshot = self._transient.is_transient(mono)

        if self.learner.is_learning:
            self.learner.update(mono)

        # Smoothed duck envelope (per-sample ramp, no hard steps)
        duck_target = self.gunshot_duck if is_gunshot else 1.0
        duck = self._duck.ramp(frames, duck_target)

        out = np.zeros_like(data, dtype=np.float64)

        if self.use_learned and self.learner.learned_mask is not None:
            self._ensure_ola(n_channels, frames)
            for ch in range(n_channels):
                out[:, ch] = self._ola[ch].process(data[:, ch].astype(np.float64))
        else:
            for ch in range(n_channels):
                bp, hp, lp = self._filter_channel(ch, data[:, ch])
                out[:, ch] = (
                    bp * self.footstep_gain
                    + (hp + lp) * self.ambient_suppress
                )

        out *= duck[:, np.newaxis]
        out = self._limiter.process(out)

        result = out[:, 0] if not stereo else out
        return result.astype(np.float32)
