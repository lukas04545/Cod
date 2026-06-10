"""
Regression suite for the footstep enhancer.  Run with:  python -m pytest
"""
import numpy as np
import pytest
from scipy import signal as sp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_processor import FootstepEnhancer, SpectralEngine

FS = 48000
HOP = 512


# ---------------------------------------------------------------------------
# Scene synthesis
# ---------------------------------------------------------------------------

def make_scene(seconds=6, steps=True, gunshots=False, seed=7):
    rng = np.random.default_rng(seed)
    n = seconds * FS
    amb = sp.lfilter(*sp.butter(2, 2000 / (FS / 2), 'low'),
                     rng.standard_normal(n) * 0.05)
    sig = amb.copy()
    step_times, shot_times = [], []
    if steps:
        bs, as_ = sp.butter(4, [250 / (FS / 2), 700 / (FS / 2)], 'band')
        for st in np.arange(0.5, seconds - 0.3, 0.6):
            i0 = int(st * FS)
            burst = sp.lfilter(bs, as_, rng.standard_normal(int(0.08 * FS)))
            sig[i0:i0 + len(burst)] += burst * np.exp(
                -np.linspace(0, 5, len(burst))) * 0.5
            step_times.append(i0)
    if gunshots:
        for st in [1.7, 3.7]:
            i0 = int(st * FS)
            crack = rng.standard_normal(int(0.12 * FS))
            sig[i0:i0 + len(crack)] += crack * np.exp(
                -np.linspace(0, 6, len(crack))) * 0.85
            shot_times.append(i0)
    return sig.astype(np.float32), step_times, amb.astype(np.float32), shot_times


def run(enh, audio):
    blocks = [enh.process(audio[i:i + HOP])
              for i in range(0, len(audio) - HOP, HOP)]
    return np.concatenate(blocks)


def contrast(sig, times):
    win = int(0.1 * FS)
    s = np.mean([np.sqrt(np.mean(sig[t:t + win] ** 2))
                 for t in times if t + win < len(sig)])
    a = np.mean([np.sqrt(np.mean(sig[t - 2 * win:t - win] ** 2))
                 for t in times if t > 2 * win])
    return s / (a + 1e-12)


def learn_from(enh, sig):
    enh.learner.start()
    for i in range(0, len(sig), HOP):
        enh.learner.update(sig[i:i + HOP])
    enh.learner.stop()
    return enh.learner.finalize()


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def test_engine_transparency():
    eng = SpectralEngine(HOP, 1, FS)
    ones = np.ones(len(eng.freqs))
    t = np.linspace(0, 1, FS, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t)
    out = np.concatenate([eng.process(sine[i:i + HOP][:, None], ones, 0.0)[0]
                          for i in range(0, FS, HOP)])[:, 0]
    err = np.max(np.abs(out[HOP:][2048:40000] - sine[:-HOP][2048:40000]))
    assert err < 5e-4  # < -66 dB


def test_partial_block():
    eng = SpectralEngine(HOP, 2, FS)
    ones = np.ones(len(eng.freqs))
    out, e, ch = eng.process(np.zeros((100, 2)), ones, 1.0)
    assert out.shape == (100, 2)
    assert len(ch) == 2


# ---------------------------------------------------------------------------
# Detection & enhancement
# ---------------------------------------------------------------------------

def test_footstep_detection_count():
    scene, step_times, _, _ = make_scene()
    stereo = np.column_stack([scene, scene])
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    detections, prev = 0, 0
    for i in range(0, len(scene) - HOP, HOP):
        enh.process(stereo[i:i + HOP])
        if enh.footstep_active == enh._hold_blocks and prev < enh.footstep_active:
            detections += 1
        prev = enh.footstep_active
    assert len(step_times) * 0.7 <= detections <= len(step_times) * 1.5


def test_contrast_improvement():
    scene, step_times, _, _ = make_scene()
    stereo = np.column_stack([scene, scene])
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    out = run(enh, stereo)[:, 0]
    assert contrast(out, [t + HOP for t in step_times]) \
        > contrast(scene, step_times) * 2


def test_gunshots_ducked_footsteps_not():
    scene, step_times, _, shot_times = make_scene(gunshots=True)
    step_times = [s for s in step_times
                  if all(abs(s - g) > 0.3 * FS for g in shot_times)]
    stereo = np.column_stack([scene, scene])
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    out = run(enh, stereo)[:, 0]
    win = int(0.1 * FS)
    shot_in = np.mean([np.sqrt(np.mean(scene[g:g + win] ** 2)) for g in shot_times])
    shot_out = np.mean([np.sqrt(np.mean(out[g + HOP:g + HOP + win] ** 2))
                        for g in shot_times])
    step_in = np.mean([np.sqrt(np.mean(scene[s:s + win] ** 2)) for s in step_times])
    step_out = np.mean([np.sqrt(np.mean(out[s + HOP:s + HOP + win] ** 2))
                        for s in step_times if s + HOP + win < len(out)])
    assert shot_out < shot_in * 0.5
    assert step_out > step_in * 0.8


def test_stereo_imaging_preserved():
    scene, _, _, _ = make_scene()
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    out = run(enh, np.column_stack([scene, np.zeros_like(scene)]))
    l = np.sqrt(np.mean(out[:, 0] ** 2))
    r = np.sqrt(np.mean(out[:, 1] ** 2))
    assert r < l * 0.01


def test_ambience_suppressed():
    _, _, amb, _ = make_scene()
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    out = run(enh, np.column_stack([amb, amb]))
    red = (np.sqrt(np.mean(out[-2 * FS:, 0] ** 2))
           / np.sqrt(np.mean(amb[-2 * FS:] ** 2)))
    assert red < 0.55  # > 5 dB reduction of constant background


def test_agc_raises_quiet_steps():
    scene, step_times, _, _ = make_scene()
    quiet = np.column_stack([scene, scene]) * 0.08
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    enh.agc_enabled = True
    out = run(enh, quiet)
    qi = np.mean([np.sqrt(np.mean(quiet[t:t + 4800, 0] ** 2)) for t in step_times])
    qo = np.mean([np.sqrt(np.mean(out[t + HOP:t + HOP + 4800, 0] ** 2))
                  for t in step_times if t + HOP + 4800 < len(out)])
    assert qo > qi * 1.5


def test_step_direction():
    scene, step_times, _, _ = make_scene()
    # Steps panned hard left
    left = np.column_stack([scene, scene * 0.1])
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    directions = []
    prev = 0
    for i in range(0, len(scene) - HOP, HOP):
        enh.process(left[i:i + HOP])
        if enh.footstep_active == enh._hold_blocks and prev < enh.footstep_active:
            directions.append(enh.last_direction)
        prev = enh.footstep_active
    assert directions, "no steps detected"
    assert np.mean(directions) < -0.3  # clearly left


# ---------------------------------------------------------------------------
# Learning, fingerprint, profiles
# ---------------------------------------------------------------------------

def footstep_tones(seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, FS, endpoint=False)
    return (0.12 * np.sin(2 * np.pi * 300 * t)
            + 0.08 * np.sin(2 * np.pi * 550 * t)
            + 0.06 * np.sin(2 * np.pi * 720 * t)
            + 0.02 * rng.standard_normal(FS)).astype(np.float32)


def test_learning_finds_peaks():
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    assert learn_from(enh, footstep_tones()) is not None
    for tgt in [300, 550, 720]:
        nearest = min(enh.learner.peak_freqs, key=lambda p: abs(p - tgt))
        assert abs(nearest - tgt) < 60


def test_fingerprint_rejects_broadband():
    """With a learned profile, a moderate broadband transient (reload,
    grenade pin) must NOT trigger the step boost, but a matching
    band-limited transient must."""
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    # Learn from band-limited bursts (footstep-like)
    rng = np.random.default_rng(3)
    bs, as_ = sp.butter(4, [250 / (FS / 2), 700 / (FS / 2)], 'band')
    train = np.zeros(4 * FS, dtype=np.float32)
    for st in np.arange(0.3, 3.7, 0.4):
        i0 = int(st * FS)
        burst = sp.lfilter(bs, as_, rng.standard_normal(int(0.08 * FS)))
        train[i0:i0 + len(burst)] += (burst * np.exp(
            -np.linspace(0, 5, len(burst))) * 0.4).astype(np.float32)
    assert learn_from(enh, train) is not None

    def count_detections(sig):
        e = FootstepEnhancer(sample_rate=FS, block_size=HOP)
        e.learner = enh.learner
        st = np.column_stack([sig, sig])
        det, prev = 0, 0
        for i in range(0, len(sig) - HOP, HOP):
            e.process(st[i:i + HOP])
            if e.footstep_active == e._hold_blocks and prev < e.footstep_active:
                det += 1
            prev = e.footstep_active
        return det

    base = (rng.standard_normal(4 * FS) * 0.02).astype(np.float32)
    base = sp.lfilter(*sp.butter(2, 2000 / (FS / 2), 'low'), base).astype(np.float32)

    # Footstep-like bursts → should be detected
    step_sig = base.copy()
    for st in [1.0, 2.0, 3.0]:
        i0 = int(st * FS)
        burst = sp.lfilter(bs, as_, rng.standard_normal(int(0.08 * FS)))
        step_sig[i0:i0 + len(burst)] += (burst * np.exp(
            -np.linspace(0, 5, len(burst))) * 0.3).astype(np.float32)

    # Broadband (flat) bursts at similar level → should be rejected
    flat_sig = base.copy()
    for st in [1.0, 2.0, 3.0]:
        i0 = int(st * FS)
        burst = rng.standard_normal(int(0.08 * FS))
        flat_sig[i0:i0 + len(burst)] += (burst * np.exp(
            -np.linspace(0, 5, len(burst))) * 0.08).astype(np.float32)

    assert count_detections(step_sig) >= 2
    assert count_detections(flat_sig) <= 1


def learn_live(enh, sig):
    """Learn through the full process() path so onsets get labelled."""
    stereo = np.column_stack([sig, sig])
    enh.learner.start()
    for i in range(0, len(sig) - HOP, HOP):
        enh.process(stereo[i:i + HOP])
    enh.learner.stop()
    return enh.learner.finalize()


def make_training_scene(seconds=8, tone_hz=1800, seed=11):
    """Step bursts + a constant moderate tone (e.g. music bed)."""
    rng = np.random.default_rng(seed)
    n = seconds * FS
    t = np.arange(n) / FS
    tone = (0.03 * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)
    sig = tone + (rng.standard_normal(n) * 0.01).astype(np.float32)
    bs, as_ = sp.butter(4, [250 / (FS / 2), 700 / (FS / 2)], 'band')
    for st in np.arange(0.5, seconds - 0.3, 0.45):
        i0 = int(st * FS)
        burst = sp.lfilter(bs, as_, rng.standard_normal(int(0.08 * FS)))
        sig[i0:i0 + len(burst)] += (burst * np.exp(
            -np.linspace(0, 5, len(burst))) * 0.4).astype(np.float32)
    return sig


def test_onset_gated_learning_excludes_background_tone():
    """A constant tone playing during learning must not enter the
    fingerprint when onset gating is active."""
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    sig = make_training_scene(tone_hz=1800)
    assert learn_live(enh, sig) is not None
    assert enh.learner.learned_from_onsets, "onset gating did not engage"
    # The 1800 Hz tone must not be a learned peak
    assert all(abs(p - 1800) > 150 for p in enh.learner.peak_freqs), \
        f"background tone leaked into fingerprint: {enh.learner.peak_freqs}"
    # The actual step band must be represented
    assert any(200 < p < 800 for p in enh.learner.peak_freqs)


def test_classifier_trained_and_discriminates():
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    learn_live(enh, make_training_scene())
    clf = enh.learner.classifier
    assert clf is not None, "classifier did not train"

    freqs = enh.learner.freqs
    # Footstep-shaped spectrum: energy concentrated 250-700 Hz
    step_mag = np.exp(-0.5 * ((freqs - 450) / 150) ** 2)
    # Broadband flat spectrum (reload click, static burst)
    flat_mag = np.ones_like(freqs) * 0.1

    p_step = clf.predict(step_mag, freqs)
    p_flat = clf.predict(flat_mag, freqs)
    assert p_step > 0.5, f"step score too low: {p_step:.2f}"
    assert p_flat < 0.5, f"flat score too high: {p_flat:.2f}"


def test_tonal_transients_rejected_after_learning():
    """A beep-like tonal burst (e.g. hitmarker at 1800 Hz) produces a flux
    onset but must be rejected by the combined classifier+fingerprint gate."""
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    learn_live(enh, make_training_scene())
    assert enh.learner.classifier is not None

    rng = np.random.default_rng(5)
    n = 4 * FS
    sig = (rng.standard_normal(n) * 0.008).astype(np.float32)
    t = np.arange(int(0.1 * FS)) / FS
    beep = (0.09 * np.sin(2 * np.pi * 1800 * t)
            * np.exp(-np.linspace(0, 4, len(t)))).astype(np.float32)
    for st in [1.0, 2.0, 3.0]:
        i0 = int(st * FS)
        sig[i0:i0 + len(beep)] += beep

    stereo = np.column_stack([sig, sig])
    det, prev = 0, 0
    for i in range(0, n - HOP, HOP):
        enh.process(stereo[i:i + HOP])
        if enh.footstep_active == enh._hold_blocks and prev < enh.footstep_active:
            det += 1
        prev = enh.footstep_active
    assert det <= 1, f"tonal beeps triggered {det} step detections"


def test_profile_roundtrip_with_classifier(tmp_path):
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    learn_live(enh, make_training_scene())
    assert enh.learner.classifier is not None
    path = str(tmp_path / "p.json")
    enh.learner.save_profile(path)

    enh2 = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    enh2.learner.load_profile(path)
    assert enh2.learner.classifier is not None
    freqs = enh2.learner.freqs
    mag = np.exp(-0.5 * ((freqs - 450) / 150) ** 2)
    assert abs(enh2.learner.classifier.predict(mag, freqs)
               - enh.learner.classifier.predict(mag, freqs)) < 1e-9
    assert enh2.learner.learned_from_onsets == enh.learner.learned_from_onsets


def test_profile_roundtrip(tmp_path):
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    assert learn_from(enh, footstep_tones()) is not None
    path = str(tmp_path / "profile.json")
    enh.learner.save_profile(path)

    enh2 = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    enh2.learner.load_profile(path)
    assert enh2.learner.learned_mask is not None
    np.testing.assert_allclose(enh2.learner.learned_mask,
                               enh.learner.learned_mask, rtol=1e-6)
    assert enh2.learner.peak_freqs == enh.learner.peak_freqs
    # Loaded profile must be usable immediately
    enh2.use_learned = True
    out = enh2.process(np.zeros((HOP, 2), dtype=np.float32))
    assert out.shape == (HOP, 2)


def test_save_without_profile_raises(tmp_path):
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    with pytest.raises(ValueError):
        enh.learner.save_profile(str(tmp_path / "x.json"))


# ---------------------------------------------------------------------------
# Cadence, sidechain, widening
# ---------------------------------------------------------------------------

def test_cadence_tracker_locks_and_anticipates():
    from audio_processor import CadenceTracker
    block_dur = HOP / FS
    ct = CadenceTracker(block_dur)
    blocks_per_step = int(0.5 / block_dur)

    # Three regular steps 0.5 s apart
    for _ in range(3):
        for _ in range(blocks_per_step):
            ct.tick()
        ct.on_step()
    assert ct.locked

    # Quarter period after the last step: not due yet
    for _ in range(blocks_per_step // 4):
        ct.tick()
    assert ct.threshold_scale() == 1.0
    # At the predicted time: due → raised sensitivity
    for _ in range(blocks_per_step - blocks_per_step // 4):
        ct.tick()
    assert ct.threshold_scale() > 1.0
    # Rhythm expires after a long silence
    for _ in range(int(2.5 / block_dur)):
        ct.tick()
    assert not ct.locked


def test_cadence_catches_quiet_steps():
    """Loud steps establish a rhythm; following quiet steps in the same
    rhythm are caught only with cadence enabled."""
    rng = np.random.default_rng(13)
    n = 8 * FS
    amb = sp.lfilter(*sp.butter(2, 2000 / (FS / 2), 'low'),
                     rng.standard_normal(n) * 0.05).astype(np.float32)
    bs, as_ = sp.butter(4, [250 / (FS / 2), 700 / (FS / 2)], 'band')

    def add_step(sig, t0, amp):
        i0 = int(t0 * FS)
        burst = sp.lfilter(bs, as_, rng.standard_normal(int(0.08 * FS)))
        sig[i0:i0 + len(burst)] += (burst * np.exp(
            -np.linspace(0, 5, len(burst))) * amp).astype(np.float32)

    sig = amb.copy()
    times = list(np.arange(0.5, 7.5, 0.5))
    for i, t0 in enumerate(times):
        add_step(sig, t0, 0.5 if i < 4 else 0.13)  # rhythm gets quiet

    def detections(cadence):
        enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
        enh.cadence_enabled = cadence
        st = np.column_stack([sig, sig])
        det, prev = 0, 0
        for i in range(0, n - HOP, HOP):
            enh.process(st[i:i + HOP])
            if enh.footstep_active == enh._hold_blocks and prev < enh.footstep_active:
                det += 1
            prev = enh.footstep_active
        return det

    with_c, without_c = detections(True), detections(False)
    assert with_c > without_c, \
        f"cadence should catch extra steps ({with_c} vs {without_c})"


def test_sidechain_ducks_out_of_band_during_steps():
    rng = np.random.default_rng(17)
    n = 6 * FS
    t = np.arange(n) / FS
    tone = (0.05 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    sig = tone + (rng.standard_normal(n) * 0.005).astype(np.float32)
    bs, as_ = sp.butter(4, [250 / (FS / 2), 700 / (FS / 2)], 'band')
    step_times = []
    for st in np.arange(0.8, 5.5, 0.7):
        i0 = int(st * FS)
        burst = sp.lfilter(bs, as_, rng.standard_normal(int(0.08 * FS)))
        sig[i0:i0 + len(burst)] += (burst * np.exp(
            -np.linspace(0, 5, len(burst))) * 0.4).astype(np.float32)
        step_times.append(i0)

    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    enh.noise_reduction = 0.0   # isolate the sidechain effect
    enh.step_sidechain = 0.6
    out = run(enh, np.column_stack([sig, sig]))[:, 0]

    def tone_rms(seg):
        b, a = sp.butter(4, [2700 / (FS / 2), 3300 / (FS / 2)], 'band')
        return np.sqrt(np.mean(sp.lfilter(b, a, seg) ** 2))

    win = int(0.1 * FS)
    during = np.mean([tone_rms(out[s + HOP:s + HOP + win]) for s in step_times])
    # 300 ms after each step: boost envelope mostly decayed
    between = np.mean([tone_rms(out[s + HOP + 3 * win:s + HOP + 4 * win])
                       for s in step_times])
    assert during < between * 0.85, \
        f"tone should duck during steps ({during:.5f} vs {between:.5f})"


def test_direction_widening():
    scene, step_times, _, _ = make_scene()
    panned = np.column_stack([scene, scene * 0.5])  # steps on the left

    def lr_ratio(widen):
        enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
        enh.direction_widen = widen
        out = run(enh, panned)
        win = int(0.1 * FS)
        l = np.mean([np.sqrt(np.mean(out[s + HOP:s + HOP + win, 0] ** 2))
                     for s in step_times if s + HOP + win < len(out)])
        r = np.mean([np.sqrt(np.mean(out[s + HOP:s + HOP + win, 1] ** 2))
                     for s in step_times if s + HOP + win < len(out)])
        return l / (r + 1e-12)

    assert lr_ratio(1.0) > lr_ratio(0.0) * 1.1, \
        "widening should exaggerate the L/R difference of left-panned steps"


# ---------------------------------------------------------------------------
# Device / sample-rate changes
# ---------------------------------------------------------------------------

def test_set_sample_rate_keeps_profile_and_processes():
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    assert learn_from(enh, footstep_tones()) is not None
    old_peaks = list(enh.learner.peak_freqs)

    enh.set_sample_rate(44100)
    assert enh.learner.sample_rate == 44100
    # Profile survives the device switch
    assert enh.learner.learned_mask is not None
    assert enh.learner.peak_freqs == old_peaks
    # Mask must boost the learned band on the new frequency grid too
    freqs = enh.learner.freqs
    band = enh.learner.learned_mask[(freqs > 280) & (freqs < 320)]
    assert band.max() > 1.5

    enh.use_learned = True
    out = enh.process(np.zeros((HOP, 2), dtype=np.float32))
    assert out.shape == (HOP, 2)

    # Same rate again: cheap reset path, still functional
    enh.set_sample_rate(44100)
    out = enh.process((np.random.default_rng(0)
                       .standard_normal((HOP, 2)) * 0.1).astype(np.float32))
    assert out.shape == (HOP, 2)
    assert np.all(np.isfinite(out))


def test_stream_engine_update_devices():
    """Regression: update_devices() used to call removed Butterworth
    internals (_design_filters/_bp_zi) and crash."""
    try:
        from stream_engine import StreamEngine
    except OSError:
        pytest.skip("PortAudio not available in this environment")
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    eng = StreamEngine(processor=enh, input_device=None, output_device=None,
                       sample_rate=FS, block_size=HOP, channels=2)
    eng.update_devices(None, None, 44100, 2)   # stream not running: no audio I/O
    assert enh.sample_rate == 44100
    out = enh.process(np.zeros((HOP, 2), dtype=np.float32))
    assert out.shape == (HOP, 2)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def test_cpu_budget():
    import time
    enh = FootstepEnhancer(sample_rate=FS, block_size=HOP)
    blk = (np.random.default_rng(0).standard_normal((HOP, 2)) * 0.2
           ).astype(np.float32)
    enh.process(blk)  # warm-up
    start = time.perf_counter()
    for _ in range(300):
        enh.process(blk)
    per_block_ms = (time.perf_counter() - start) / 300 * 1000
    assert per_block_ms < 5  # budget is 10.7 ms
