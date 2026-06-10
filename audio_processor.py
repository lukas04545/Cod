import json
import numpy as np
from scipy.signal import find_peaks, savgol_filter, windows
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

    # ------------------------------------------------------------------
    # Profile persistence (per game / map)

    def save_profile(self, path: str) -> None:
        if self.learned_mask is None:
            raise ValueError("No learned profile to save")
        profile = {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "mask": self.learned_mask.tolist(),
            "peak_freqs": self.peak_freqs,
            "avg_spectrum": self.avg_spectrum.tolist(),
        }
        with open(path, "w") as f:
            json.dump(profile, f)

    def load_profile(self, path: str) -> None:
        with open(path) as f:
            profile = json.load(f)
        # Re-interpolate onto our own bins in case the profile was saved
        # with a different FFT size or sample rate
        src_freqs = np.fft.rfftfreq(profile["n_fft"],
                                    1.0 / profile["sample_rate"])
        self.learned_mask = np.interp(self.freqs, src_freqs,
                                      np.asarray(profile["mask"]))
        self.peak_freqs = [float(p) for p in profile["peak_freqs"]]
        with self._lock:
            self.avg_spectrum = np.interp(self.freqs, src_freqs,
                                          np.asarray(profile["avg_spectrum"]))
        self.mask_version += 1


# ---------------------------------------------------------------------------
# Envelope / dynamics helpers
# ---------------------------------------------------------------------------

class GainSmoother:
    """
    Exponentially smoothed gain envelope with independent rise and fall
    time constants.  Eliminates clicks/pumping from per-block gain steps.
    """

    def __init__(self, sample_rate: int, rise_ms: float, fall_ms: float,
                 initial: float = 1.0):
        self._rise = float(np.exp(-1.0 / (sample_rate * rise_ms / 1000.0)))
        self._fall = float(np.exp(-1.0 / (sample_rate * fall_ms / 1000.0)))
        self._current = initial

    def ramp(self, frames: int, target: float) -> np.ndarray:
        """Return per-sample gain ramp moving toward target."""
        coef = self._rise if target > self._current else self._fall
        n = np.arange(1, frames + 1, dtype=np.float64)
        gains = target + (self._current - target) * coef ** n
        self._current = float(gains[-1])
        return gains


class SoftLimiter:
    """Peak limiter with smoothed gain (fast engage, slow recover)."""

    def __init__(self, sample_rate: int, threshold: float = 0.90):
        self.threshold = threshold
        self._smoother = GainSmoother(sample_rate, rise_ms=120.0, fall_ms=1.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) + 1e-12
        target = 1.0 if peak <= self.threshold else self.threshold / peak
        gains = self._smoother.ramp(x.shape[0], target)
        if x.ndim == 2:
            gains = gains[:, np.newaxis]
        return np.clip(x * gains, -1.0, 1.0)


class AutoGainControl:
    """
    Slow loudness normalisation: brings distant quiet footsteps up to a
    comfortable level without pumping. Holds gain during near-silence so
    the noise floor isn't dragged up between sounds.
    """

    def __init__(self, sample_rate: int, target_rms: float = 0.10,
                 max_gain: float = 8.0, min_gain: float = 0.5):
        self.target_rms = target_rms
        self.max_gain = max_gain
        self.min_gain = min_gain
        self._smoother = GainSmoother(sample_rate, rise_ms=500.0, fall_ms=150.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms < 1e-4:
            # Near-silence: hold the current gain
            gains = self._smoother.ramp(x.shape[0], self._smoother._current)
        else:
            desired = float(np.clip(self.target_rms / rms, self.min_gain,
                                    self.max_gain))
            gains = self._smoother.ramp(x.shape[0], desired)
        if x.ndim == 2:
            gains = gains[:, np.newaxis]
        return x * gains


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class TransientDetector:
    """
    Gunshot/explosion detector.  A naive RMS-jump check also fires on
    footsteps (they are transients too) — which would duck the very sounds
    we want to boost.  A gunshot must satisfy ALL of:

      1. RMS jump over recent history (> threshold_db)
      2. loud in absolute terms (rms > loud_floor)
      3. broadband: significant energy above hf_cutoff Hz (the "crack");
         footsteps live almost entirely below ~1 kHz.

    Once triggered, stays hot for hold_blocks so the whole tail is ducked.
    """

    def __init__(self, sample_rate: int = 48000, window: int = 20,
                 threshold_db: float = GUNSHOT_TRANSIENT_DB,
                 hold_blocks: int = 12, loud_floor: float = 0.12,
                 hf_cutoff: float = 1500.0, hf_ratio: float = 0.30):
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.hold_blocks = hold_blocks
        self.loud_floor = loud_floor
        self.hf_cutoff = hf_cutoff
        self.hf_ratio = hf_ratio
        self._history: deque[float] = deque(maxlen=window)
        self._hold = 0
        self._lock = threading.Lock()

    def _hf_fraction(self, block: np.ndarray) -> float:
        mag2 = np.abs(np.fft.rfft(block)) ** 2
        freqs = np.fft.rfftfreq(len(block), 1.0 / self.sample_rate)
        total = float(np.sum(mag2)) + 1e-12
        return float(np.sum(mag2[freqs >= self.hf_cutoff])) / total

    def is_transient(self, block: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-12)
        with self._lock:
            self._history.append(rms)
            if len(self._history) < 5:
                return False
            mean_rms = float(np.mean(list(self._history)[:-1]))
            jumped = 20 * np.log10(rms / (mean_rms + 1e-12)) > self.threshold_db
            if jumped and rms > self.loud_floor \
                    and self._hf_fraction(block) > self.hf_ratio:
                self._hold = self.hold_blocks
            if self._hold > 0:
                self._hold -= 1
                return True
            return False


class FootstepDetector:
    """
    Detects footstep *events* via spectral flux in the footstep band,
    normalised by the long-term average band energy.  Measured on synthetic
    CoD-like scenes: ambience flux never exceeds ~1.1× mean energy, while
    step onsets reach 1.6–7.5× — so the threshold sits between the two.

    A refractory period prevents one step from double-triggering
    (real steps are ≥ ~300 ms apart even when sprinting).

    sensitivity: 1 (strict) … 10 (hair-trigger).
    """

    REFRACTORY_S = 0.18

    def __init__(self, sensitivity: float = 5.0, history_blocks: int = 90,
                 block_duration: float = 512 / 48000):
        self.sensitivity = sensitivity
        self._prev_energy = 0.0
        self._energy_hist: deque[float] = deque(maxlen=history_blocks)
        self._refractory_blocks = max(1, int(self.REFRACTORY_S / block_duration))
        self._cooldown = 0

    def update(self, band_energy: float) -> bool:
        flux = max(0.0, band_energy - self._prev_energy)
        self._prev_energy = band_energy
        self._energy_hist.append(band_energy)

        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if len(self._energy_hist) < 15:
            return False

        mean_energy = float(np.mean(self._energy_hist)) + 1e-9
        c = 6.5 / max(self.sensitivity, 0.5)   # sens 5 → flux must be > 1.3× mean energy
        if flux > c * mean_energy and band_energy > 1.2 * mean_energy:
            self._cooldown = self._refractory_blocks
            return True
        return False


# ---------------------------------------------------------------------------
# Spectral engine (OLA STFT, linked-gain stereo)
# ---------------------------------------------------------------------------

class SpectralEngine:
    """
    Windowed overlap-add STFT processor for N channels.

    Per-bin gains are computed ONCE from the channel-averaged spectrum and
    applied identically to every channel — this preserves interaural level
    differences exactly, so footstep direction is not smeared.

    Gain chain per bin:
      1. Adaptive noise-floor removal (minimum statistics + over-subtraction)
         → strips constant ambience: wind, music beds, electrical hum.
      2. Shape mask (fixed bandpass shape or learned footstep fingerprint).
      3. Temporal gain smoothing → suppresses musical-noise artifacts.

    Adds hop-size samples latency (~10.7 ms at 48 kHz / 512).
    """

    NOISE_ATTACK = 0.005     # per-block adaptation toward background level
    NOISE_DRIFT = 1.0005     # slow upward drift while gated (handles level rises)
    NOISE_GATE = 2.5         # bins louder than gate×floor are "events", not noise
    GAIN_FLOOR = 0.10        # never attenuate a bin below 10 % (less musical noise)
    # Asymmetric temporal gain smoothing: gains open almost instantly so the
    # sharp attack of a step is not dulled, but close slowly so noise between
    # events doesn't flutter (musical noise).
    GAIN_RISE = 0.30         # state weight when gain is increasing (fast open)
    GAIN_FALL = 0.75         # state weight when gain is decreasing (slow close)

    def __init__(self, hop: int, n_channels: int, sample_rate: int):
        self.hop = hop
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.win_len = hop * 2
        self.window = windows.hann(self.win_len, sym=False)
        self.freqs = np.fft.rfftfreq(self.win_len, 1.0 / sample_rate)

        self._in_bufs = [np.zeros(self.win_len) for _ in range(n_channels)]
        self._out_bufs = [np.zeros(self.win_len) for _ in range(n_channels)]
        self._noise: np.ndarray | None = None
        self._gain_state: np.ndarray | None = None
        self.last_clean_mag: np.ndarray | None = None

    def process(self, data: np.ndarray, shape_mask: np.ndarray,
                nr_strength: float, band_boost: float = 1.0
                ) -> tuple[np.ndarray, float, list[float]]:
        """
        Args:
            data:        (hop, n_channels) input block
            shape_mask:  per-bin target gains, len == len(self.freqs)
            nr_strength: noise-reduction over-subtraction (0 = off, 1 = normal)
            band_boost:  extra gain applied only where the mask is strong
                         (footstep band) — boosting the band instead of the
                         whole signal keeps residual noise from pumping up
                         with each detected step.

        Returns:
            (processed block (hop, n_channels),
             footstep-band energy scalar,
             per-channel band energies — for direction estimation)
        """
        n = data.shape[0]
        if n < self.hop:
            padded = np.zeros((self.hop, data.shape[1]))
            padded[:n] = data
            out, e, ch_e = self.process(padded, shape_mask, nr_strength,
                                        band_boost)
            return out[:n], e, ch_e

        specs = []
        for ch in range(self.n_channels):
            buf = self._in_bufs[ch]
            buf[:-self.hop] = buf[self.hop:]
            buf[-self.hop:] = data[:, ch]
            specs.append(np.fft.rfft(buf * self.window))

        mono_mag = np.mean([np.abs(s) for s in specs], axis=0)

        # --- adaptive noise floor (gated average) ---
        # Bins near the current floor estimate are treated as background and
        # averaged in; bins far above it (footsteps, voices, shots) are
        # excluded so events never inflate the floor.  Converges to the true
        # mean background level — unlike minimum statistics, which tracks the
        # lower envelope and under-subtracts by 2-3×.
        if self._noise is None:
            self._noise = mono_mag.copy()
        else:
            background = mono_mag < self.NOISE_GATE * self._noise
            self._noise = np.where(
                background,
                (1 - self.NOISE_ATTACK) * self._noise + self.NOISE_ATTACK * mono_mag,
                self._noise * self.NOISE_DRIFT,
            )

        beta = 1.7 * nr_strength
        clean = np.maximum(mono_mag - beta * self._noise,
                           self.GAIN_FLOOR * mono_mag)
        nr_gain = clean / (mono_mag + 1e-12)

        # Band-limited boost: scale only where the mask is strong
        w = shape_mask / (shape_mask.max() + 1e-12)
        boosted_mask = shape_mask * (1.0 + (band_boost - 1.0) * w)

        total_gain = nr_gain * boosted_mask

        # --- asymmetric temporal smoothing (anti musical-noise) ---
        if self._gain_state is None or len(self._gain_state) != len(total_gain):
            self._gain_state = total_gain
        else:
            smooth = np.where(total_gain > self._gain_state,
                              self.GAIN_RISE, self.GAIN_FALL)
            self._gain_state = (smooth * self._gain_state
                                + (1 - smooth) * total_gain)
        g = self._gain_state

        # --- apply linked gains, overlap-add ---
        out = np.zeros((self.hop, self.n_channels))
        for ch in range(self.n_channels):
            y = np.fft.irfft(specs[ch] * g, n=self.win_len)
            ob = self._out_bufs[ch]
            ob[:-self.hop] = ob[self.hop:]
            ob[-self.hop:] = 0.0
            ob += y
            out[:, ch] = ob[:self.hop]

        # Footstep-band energy of the cleaned signal, weighted by the mask
        band_energy = float(np.sum(mono_mag * nr_gain * w))
        ch_energies = [float(np.sum(np.abs(specs[ch]) * nr_gain * w))
                       for ch in range(self.n_channels)]

        # Cleaned mono spectrum, used for fingerprint matching upstream
        self.last_clean_mag = mono_mag * nr_gain

        return out, band_energy, ch_energies


# ---------------------------------------------------------------------------
# Main enhancer
# ---------------------------------------------------------------------------

def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class FootstepEnhancer:
    """
    Full enhancement chain (everything runs in the spectral engine):

      input → STFT → adaptive noise removal → shape mask (fixed band or
      learned fingerprint) → iSTFT → footstep-event boost → gunshot duck
      → optional auto-gain → soft limiter → output

    A spectral-flux footstep detector fires on each step onset and opens a
    fast-attack/slow-release boost envelope, making individual steps pop
    out of the mix. All per-bin gains are linked across channels so stereo
    directionality is preserved exactly.
    """

    FOOTSTEP_LOW_HZ = 150
    FOOTSTEP_HIGH_HZ = 900

    def __init__(self, sample_rate: int = 48000, block_size: int = 512):
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Tunable from GUI
        self.footstep_gain: float = 3.0       # in-band gain (fixed mode shape)
        self.ambient_suppress: float = 0.25   # out-of-band gain
        self.gunshot_duck: float = 0.10
        self.noise_reduction: float = 1.0     # 0 = off … 1 = full
        self.footstep_boost: float = 2.5      # event boost multiplier
        self.detect_sensitivity: float = 5.0  # 1 … 10
        self.agc_enabled: bool = False
        self.enabled: bool = True
        self.use_learned: bool = False

        self.learner = FrequencyLearner(
            sample_rate=sample_rate,
            n_fft=max(block_size * 8, 4096),
        )

        self._gunshot = TransientDetector(sample_rate=sample_rate)
        self._footstep = FootstepDetector(
            sensitivity=self.detect_sensitivity,
            block_duration=block_size / sample_rate,
        )
        self._duck = GainSmoother(sample_rate, rise_ms=220.0, fall_ms=4.0)
        self._agc = AutoGainControl(sample_rate)
        self._limiter = SoftLimiter(sample_rate)

        self._engine: SpectralEngine | None = None
        self._mask_cache: np.ndarray | None = None
        self._mask_key: tuple | None = None

        # Per-block smoothed band-boost scalar (applied in spectral domain)
        self._boost_state: float = 1.0
        block_s = block_size / sample_rate
        self._boost_rise = float(np.exp(-block_s / 0.010))   # ~10 ms attack
        self._boost_fall = float(np.exp(-block_s / 0.300))   # ~300 ms release

        # Fingerprint matching (rejects non-footstep transients)
        self.fingerprint_gate: bool = True
        self.fingerprint_threshold: float = 0.60
        self._fp_cache: np.ndarray | None = None
        self._fp_region: np.ndarray | None = None
        self._fp_version: int = -1

        # GUI-readable state
        self.footstep_active: int = 0          # > 0 for ~150 ms after a step
        self.last_direction: float = 0.0       # -1 = left … +1 = right
        self._hold_blocks = max(1, int(0.15 * sample_rate / block_size))

    # ------------------------------------------------------------------
    # Shape mask construction (cached)

    def _shape_mask(self, freqs: np.ndarray) -> np.ndarray:
        key = (
            self.use_learned,
            self.learner.mask_version,
            round(self.footstep_gain, 3),
            round(self.ambient_suppress, 3),
            len(freqs),
        )
        if key == self._mask_key and self._mask_cache is not None:
            return self._mask_cache

        if self.use_learned and self.learner.learned_mask is not None:
            mask = np.interp(freqs, self.learner.freqs, self.learner.learned_mask)
        else:
            # Raised-cosine bandpass: smooth edges avoid ringing
            lo_edge = _smoothstep((freqs - (self.FOOTSTEP_LOW_HZ - 50))
                                  / 100.0)
            hi_edge = 1.0 - _smoothstep((freqs - (self.FOOTSTEP_HIGH_HZ - 100))
                                        / 300.0)
            band = lo_edge * hi_edge
            mask = self.ambient_suppress + (self.footstep_gain
                                            - self.ambient_suppress) * band

        self._mask_cache = mask
        self._mask_key = key
        return mask

    def _ensure_engine(self, hop: int, n_channels: int) -> None:
        if (self._engine is None
                or self._engine.hop != hop
                or self._engine.n_channels != n_channels):
            self._engine = SpectralEngine(hop, n_channels, self.sample_rate)
            self._mask_key = None  # bins changed → rebuild mask
            self._fp_version = -1

    # ------------------------------------------------------------------
    # Footstep fingerprint (cached unit vector of the learned spectrum)

    def _fingerprint(self, freqs: np.ndarray) -> np.ndarray | None:
        if self.learner.learned_mask is None:
            return None
        if self._fp_version == self.learner.mask_version \
                and self._fp_cache is not None:
            return self._fp_cache

        fp = np.interp(freqs, self.learner.freqs, self.learner.avg_spectrum)
        region = (freqs >= 80.0) & (freqs <= 2000.0)
        fp = fp * region
        norm = float(np.linalg.norm(fp))
        if norm < 1e-9:
            self._fp_cache = None
        else:
            self._fp_cache = fp / norm
        self._fp_region = region
        self._fp_version = self.learner.mask_version
        return self._fp_cache

    def _matches_fingerprint(self, clean_mag: np.ndarray,
                             fp: np.ndarray) -> bool:
        cur = clean_mag * self._fp_region
        norm = float(np.linalg.norm(cur))
        if norm < 1e-9:
            return False
        similarity = float(np.dot(cur / norm, fp))
        return similarity >= self.fingerprint_threshold

    # ------------------------------------------------------------------
    # Main process

    def process(self, indata: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return indata.astype(np.float32)

        stereo = indata.ndim == 2
        data = (indata if stereo else indata[:, np.newaxis]).astype(np.float64)
        frames, n_channels = data.shape

        self._ensure_engine(frames if frames >= 64 else self.block_size,
                            n_channels)

        mono = data.mean(axis=1)
        is_gunshot = self._gunshot.is_transient(mono)

        if self.learner.is_learning:
            self.learner.update(mono)

        # --- spectral processing (boost from the previous block's detection;
        #     one-block lag is far below the boost envelope's time constants) ---
        mask = self._shape_mask(self._engine.freqs)
        out, band_energy, ch_energies = self._engine.process(
            data, mask, self.noise_reduction, band_boost=self._boost_state)

        # --- footstep event detection (suppressed during gunshots) ---
        self._footstep.sensitivity = self.detect_sensitivity
        stepped = (not is_gunshot) and self._footstep.update(band_energy)

        # Fingerprint gate: with a learned profile, the onset spectrum must
        # resemble actual footsteps — rejects reloads, grenade pins, etc.
        if stepped and self.fingerprint_gate:
            fp = self._fingerprint(self._engine.freqs)
            if fp is not None and self._engine.last_clean_mag is not None:
                stepped = self._matches_fingerprint(
                    self._engine.last_clean_mag, fp)

        if stepped:
            self.footstep_active = self._hold_blocks
            if len(ch_energies) >= 2:
                left, right = ch_energies[0], ch_energies[-1]
                self.last_direction = (right - left) / (right + left + 1e-12)
            else:
                self.last_direction = 0.0
        elif self.footstep_active > 0:
            self.footstep_active -= 1

        # Smoothed band-boost scalar for the next block
        boost_target = self.footstep_boost if self.footstep_active > 0 else 1.0
        coef = self._boost_rise if boost_target > self._boost_state \
            else self._boost_fall
        self._boost_state = coef * self._boost_state + (1 - coef) * boost_target

        duck_target = self.gunshot_duck if is_gunshot else 1.0
        duck = self._duck.ramp(frames, duck_target)

        out *= duck[:, np.newaxis]

        if self.agc_enabled:
            out = self._agc.process(out)

        out = self._limiter.process(out)

        result = out[:, 0] if not stereo else out
        return result.astype(np.float32)
