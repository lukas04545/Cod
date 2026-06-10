"""
CoD Footstep Enhancer — GUI entry point.

Requirements:
  pip install sounddevice numpy scipy PyQt5 pyqtgraph

Audio routing (Windows):  Install VB-Cable or Voicemeeter.
  Game  →  VB-Cable Input  →  this app (pick "CABLE Output" as Input here)
  This app output  →  headphones / speakers

Audio routing (Linux):  Use PulseAudio/PipeWire virtual sinks.
"""

import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QSlider, QVBoxLayout,
    QWidget, QCheckBox,
)
from PyQt5.QtGui import QPalette, QColor
import pyqtgraph as pg

from audio_processor import FootstepEnhancer
from device_manager import input_devices, output_devices
from stream_engine import StreamEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ORANGE = "#ff6600"
GREEN  = "#00cc66"
RED    = "#ff4444"
GREY   = "#888888"


def _labeled_slider(label: str, lo: int, hi: int, default: int):
    """Return (outer QWidget, QSlider)."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)

    title = QLabel(label)
    title.setAlignment(Qt.AlignCenter)
    lay.addWidget(title)

    sl = QSlider(Qt.Horizontal)
    sl.setMinimum(lo)
    sl.setMaximum(hi)
    sl.setValue(default)
    lay.addWidget(sl)

    val_lbl = QLabel(str(default))
    val_lbl.setAlignment(Qt.AlignCenter)
    lay.addWidget(val_lbl)

    sl.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
    return w, sl


# ---------------------------------------------------------------------------
# Spectrum widget
# ---------------------------------------------------------------------------

class SpectrumWidget(pg.PlotWidget):
    """Shows learned average spectrum + amplification mask."""

    def __init__(self, freqs: np.ndarray):
        super().__init__()
        self.freqs = freqs
        self.setBackground("#1a1a1a")
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Amplitude")
        self.setXRange(20, 8000)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMinimumHeight(180)

        self._spectrum_curve = self.plot(pen=pg.mkPen("#00aaff", width=1.5), name="Captured spectrum")
        self._mask_curve     = self.plot(pen=pg.mkPen(ORANGE, width=2.0),    name="Learned mask")
        self._peak_lines: list[pg.InfiniteLine] = []

        legend = self.addLegend(offset=(10, 10))
        legend.addItem(self._spectrum_curve, "Captured spectrum")
        legend.addItem(self._mask_curve,     "Amplification mask")

    def update_spectrum(self, avg_spectrum: np.ndarray, mask: np.ndarray | None,
                        peak_freqs: list[float]) -> None:
        norm = avg_spectrum.max() + 1e-9
        self._spectrum_curve.setData(self.freqs, avg_spectrum / norm)

        if mask is not None:
            norm_mask = mask.max() + 1e-9
            self._mask_curve.setData(self.freqs, mask / norm_mask)
        else:
            self._mask_curve.setData([], [])

        for line in self._peak_lines:
            self.removeItem(line)
        self._peak_lines.clear()

        for f in peak_freqs:
            line = pg.InfiniteLine(
                pos=f, angle=90,
                pen=pg.mkPen(GREEN, width=1, style=Qt.DashLine),
                label=f"{f:.0f}Hz",
                labelOpts={"color": GREEN, "position": 0.95},
            )
            self.addItem(line)
            self._peak_lines.append(line)

    def clear_all(self) -> None:
        self._spectrum_curve.setData([], [])
        self._mask_curve.setData([], [])
        for line in self._peak_lines:
            self.removeItem(line)
        self._peak_lines.clear()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

LEARN_SECONDS = 8   # how long the auto-stop timer runs


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CoD Footstep Enhancer — AI Audio Filter")
        self.setMinimumWidth(720)

        self._processor = FootstepEnhancer()
        self._engine: StreamEngine | None = None
        self._learn_ticks = 0

        self._build_ui()
        self._apply_dark_theme()

        # VU meter refresh (20 fps)
        self._vu_timer = QTimer(self)
        self._vu_timer.setInterval(50)
        self._vu_timer.timeout.connect(self._update_vu)
        self._vu_timer.start()

        # Learning countdown + live spectrum update (4 fps)
        self._learn_timer = QTimer(self)
        self._learn_timer.setInterval(250)
        self._learn_timer.timeout.connect(self._learn_tick)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # Header
        hdr = QLabel("CoD Footstep Enhancer")
        hdr.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {ORANGE};")
        hdr.setAlignment(Qt.AlignCenter)
        root.addWidget(hdr)

        sub = QLabel("Learns exact footstep frequencies · Ducks gunshots · Suppresses ambience")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(sub)

        # --- Devices ---
        dev_grp = QGroupBox("Audio Devices")
        dev_lay = QVBoxLayout(dev_grp)

        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input (game audio):"))
        self._in_combo = QComboBox()
        in_row.addWidget(self._in_combo, 1)
        dev_lay.addLayout(in_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output (headphones):"))
        self._out_combo = QComboBox()
        out_row.addWidget(self._out_combo, 1)
        dev_lay.addLayout(out_row)

        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.clicked.connect(self._populate_devices)
        dev_lay.addWidget(refresh_btn)

        root.addWidget(dev_grp)
        self._populate_devices()

        # --- Controls ---
        ctrl_grp = QGroupBox("Enhancement Controls")
        ctrl_outer = QVBoxLayout(ctrl_grp)

        row1 = QHBoxLayout()
        self._gain_w, self._gain_sl = _labeled_slider("Footstep Gain\n(×0.1)", 10, 120, 30)
        self._gain_sl.valueChanged.connect(lambda v: setattr(self._processor, "footstep_gain", v * 0.1))
        row1.addWidget(self._gain_w)

        self._amb_w, self._amb_sl = _labeled_slider("Ambient Suppress\n(% kept)", 0, 100, 25)
        self._amb_sl.valueChanged.connect(lambda v: setattr(self._processor, "ambient_suppress", v / 100))
        row1.addWidget(self._amb_w)

        self._duck_w, self._duck_sl = _labeled_slider("Gunshot Duck\n(% kept)", 0, 50, 10)
        self._duck_sl.valueChanged.connect(lambda v: setattr(self._processor, "gunshot_duck", v / 100))
        row1.addWidget(self._duck_w)
        ctrl_outer.addLayout(row1)

        row2 = QHBoxLayout()
        self._nr_w, self._nr_sl = _labeled_slider("Noise Reduction\n(%)", 0, 100, 100)
        self._nr_sl.valueChanged.connect(lambda v: setattr(self._processor, "noise_reduction", v / 100))
        row2.addWidget(self._nr_w)

        self._boost_w, self._boost_sl = _labeled_slider("Step Boost\n(×0.1)", 10, 60, 25)
        self._boost_sl.valueChanged.connect(lambda v: setattr(self._processor, "footstep_boost", v * 0.1))
        row2.addWidget(self._boost_w)

        self._sens_w, self._sens_sl = _labeled_slider("Detect Sensitivity\n(1–10)", 1, 10, 5)
        self._sens_sl.valueChanged.connect(lambda v: setattr(self._processor, "detect_sensitivity", float(v)))
        row2.addWidget(self._sens_w)
        ctrl_outer.addLayout(row2)

        opt_row = QHBoxLayout()
        self._agc_cb = QCheckBox("Auto volume (brings distant quiet steps up)")
        self._agc_cb.stateChanged.connect(lambda s: setattr(self._processor, "agc_enabled", bool(s)))
        opt_row.addWidget(self._agc_cb)

        opt_row.addStretch()

        self._step_indicator = QLabel("● STEP")
        self._step_indicator.setStyleSheet("color: #333; font-weight: bold; font-size: 14px;")
        opt_row.addWidget(self._step_indicator)
        ctrl_outer.addLayout(opt_row)

        root.addWidget(ctrl_grp)

        # --- Learning section ---
        learn_grp = QGroupBox("Frequency Learning  (AI adaptive mode)")
        learn_lay = QVBoxLayout(learn_grp)

        tip = QLabel(
            "HOW TO USE:  Start the stream, enter a game, walk around on different surfaces "
            f"for ~{LEARN_SECONDS}s, then click  \"Learn Footsteps\".  "
            "The app captures only moderate-energy sounds (skips silence & gunshots), "
            "finds the exact frequency peaks, and builds a custom amplification mask."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #bbb; font-size: 11px;")
        learn_lay.addWidget(tip)

        btn_row = QHBoxLayout()
        self._learn_btn = QPushButton("Learn Footsteps")
        self._learn_btn.setFixedHeight(34)
        self._learn_btn.setStyleSheet(
            f"QPushButton {{ background: #1a5c1a; color: white; font-weight: bold; border-radius:5px; }}"
            f"QPushButton:hover {{ background: #227722; }}"
        )
        self._learn_btn.clicked.connect(self._on_learn)
        btn_row.addWidget(self._learn_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setFixedHeight(34)
        self._reset_btn.clicked.connect(self._on_reset_learn)
        btn_row.addWidget(self._reset_btn)

        self._use_learned_cb = QCheckBox("Use learned frequencies")
        self._use_learned_cb.setEnabled(False)
        self._use_learned_cb.stateChanged.connect(self._on_use_learned)
        btn_row.addWidget(self._use_learned_cb)
        learn_lay.addLayout(btn_row)

        profile_row = QHBoxLayout()
        save_btn = QPushButton("Save Profile…")
        save_btn.clicked.connect(self._on_save_profile)
        profile_row.addWidget(save_btn)
        load_btn = QPushButton("Load Profile…")
        load_btn.clicked.connect(self._on_load_profile)
        profile_row.addWidget(load_btn)
        profile_row.addStretch()
        learn_lay.addLayout(profile_row)

        self._learn_status = QLabel("Status: not started")
        self._learn_status.setStyleSheet(f"color: {GREY};")
        learn_lay.addWidget(self._learn_status)

        self._peak_label = QLabel("Detected peaks: —")
        self._peak_label.setStyleSheet("color: #bbb; font-size: 11px;")
        learn_lay.addWidget(self._peak_label)

        # Spectrum plot
        self._spectrum_widget = SpectrumWidget(self._processor.learner.freqs)
        learn_lay.addWidget(self._spectrum_widget)

        root.addWidget(learn_grp)

        # --- VU meters ---
        vu_grp = QGroupBox("Level Meters")
        vu_lay = QVBoxLayout(vu_grp)

        for label, attr in [("IN ", "_vu_in"), ("OUT", "_vu_out")]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setTextVisible(False)
            color = "#00aaff" if attr == "_vu_in" else ORANGE
            bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
            setattr(self, attr, bar)
            row.addWidget(bar, 1)
            vu_lay.addLayout(row)

        root.addWidget(vu_grp)

        # --- Bottom row ---
        btm = QHBoxLayout()

        self._enable_cb = QCheckBox("Processing enabled")
        self._enable_cb.setChecked(True)
        self._enable_cb.stateChanged.connect(lambda s: setattr(self._processor, "enabled", bool(s)))
        btm.addWidget(self._enable_cb)

        btm.addStretch()

        self._status_lbl = QLabel("Stopped")
        self._status_lbl.setStyleSheet(f"color: {GREY};")
        btm.addWidget(self._status_lbl)

        self._toggle_btn = QPushButton("Start")
        self._toggle_btn.setFixedHeight(40)
        self._toggle_btn.setFixedWidth(110)
        self._toggle_btn.setStyleSheet(
            f"QPushButton {{ background: #cc4400; color: white; font-weight: bold; border-radius:6px; }}"
            f"QPushButton:hover {{ background: {ORANGE}; }}"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        btm.addWidget(self._toggle_btn)

        root.addLayout(btm)

    def _populate_devices(self) -> None:
        self._in_combo.clear()
        self._out_combo.clear()
        try:
            for d in input_devices():
                self._in_combo.addItem(f"[{d.index}] {d.name}", d.index)
            for d in output_devices():
                self._out_combo.addItem(f"[{d.index}] {d.name}", d.index)
        except Exception as exc:
            self._status_lbl.setText(f"Device error: {exc}")

    # ------------------------------------------------------------------
    # Stream start / stop
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        if self._engine and self._engine.is_running():
            self._engine.stop()
            self._toggle_btn.setText("Start")
            self._status_lbl.setText("Stopped")
            self._status_lbl.setStyleSheet(f"color: {GREY};")
        else:
            in_idx  = self._in_combo.currentData()
            out_idx = self._out_combo.currentData()
            if in_idx is None or out_idx is None:
                self._status_lbl.setText("No device selected")
                return
            try:
                if self._engine is None:
                    self._engine = StreamEngine(
                        processor=self._processor,
                        input_device=in_idx,
                        output_device=out_idx,
                        sample_rate=48000,
                        block_size=512,
                        channels=2,
                    )
                else:
                    self._engine.update_devices(in_idx, out_idx, 48000, 2)
                self._engine.start()
                self._toggle_btn.setText("Stop")
                self._status_lbl.setText("Running …")
                self._status_lbl.setStyleSheet(f"color: {GREEN};")
            except Exception as exc:
                self._status_lbl.setText(f"Error: {exc}")
                self._status_lbl.setStyleSheet(f"color: {RED};")

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _on_learn(self) -> None:
        learner = self._processor.learner

        if learner.is_learning:
            # Manual stop → finalize immediately
            self._stop_learning()
            return

        if not (self._engine and self._engine.is_running()):
            self._learn_status.setText("Start the audio stream first.")
            self._learn_status.setStyleSheet(f"color: {RED};")
            return

        # Start learning
        learner.start()
        self._learn_ticks = LEARN_SECONDS * 4   # 250 ms ticks
        self._learn_btn.setText(f"Stop Learning  ({LEARN_SECONDS}s)")
        self._learn_btn.setStyleSheet(
            "QPushButton { background: #7a2200; color: white; font-weight: bold; border-radius:5px; }"
        )
        self._use_learned_cb.setEnabled(False)
        self._learn_status.setText(f"Listening …  0 frames captured")
        self._learn_status.setStyleSheet(f"color: {ORANGE};")
        self._learn_timer.start()

    def _learn_tick(self) -> None:
        """Called every 250 ms while learning."""
        learner = self._processor.learner
        frames = learner.frames_collected
        elapsed = LEARN_SECONDS - (self._learn_ticks * 0.25)

        # Live spectrum preview
        avg = learner.get_avg_spectrum()
        self._spectrum_widget.update_spectrum(avg, None, [])

        self._learn_status.setText(
            f"Listening …  {frames} frames captured  ({elapsed:.1f}s / {LEARN_SECONDS}s)"
        )

        self._learn_ticks -= 1
        if self._learn_ticks <= 0:
            self._stop_learning()

    def _stop_learning(self) -> None:
        self._learn_timer.stop()
        self._processor.learner.stop()
        self._learn_btn.setText("Learn Footsteps")
        self._learn_btn.setStyleSheet(
            "QPushButton { background: #1a5c1a; color: white; font-weight: bold; border-radius:5px; }"
            "QPushButton:hover { background: #227722; }"
        )

        mask = self._processor.learner.finalize()
        if mask is None:
            frames = self._processor.learner.frames_collected
            self._learn_status.setText(
                f"Not enough data ({frames} frames). Walk around longer and try again."
            )
            self._learn_status.setStyleSheet(f"color: {RED};")
            return

        learner = self._processor.learner
        peaks = learner.peak_freqs
        self._spectrum_widget.update_spectrum(learner.avg_spectrum, mask, peaks)

        peak_str = ",  ".join(f"{f:.0f} Hz" for f in peaks[:10])
        self._peak_label.setText(f"Learned peaks:  {peak_str}")

        if learner.learned_from_onsets:
            source = (f"{learner.onset_frames_collected} step-onset frames "
                      f"(clean fingerprint)")
        else:
            source = (f"{learner.frames_collected} frames "
                      f"(no clear step onsets — walk more during learning)")
        clf_note = " AI step classifier trained." if learner.classifier else ""
        self._learn_status.setText(
            f"Done! {len(peaks)} peaks from {source}.{clf_note} "
            f"Enable \"Use learned frequencies\" to activate."
        )
        self._learn_status.setStyleSheet(f"color: {GREEN};")
        self._use_learned_cb.setEnabled(True)

    def _on_reset_learn(self) -> None:
        self._learn_timer.stop()
        self._processor.learner.reset()
        self._processor.use_learned = False
        self._use_learned_cb.setChecked(False)
        self._use_learned_cb.setEnabled(False)
        self._spectrum_widget.clear_all()
        self._peak_label.setText("Detected peaks: —")
        self._learn_status.setText("Reset. Ready to learn again.")
        self._learn_status.setStyleSheet(f"color: {GREY};")
        self._learn_btn.setText("Learn Footsteps")
        self._learn_btn.setStyleSheet(
            "QPushButton { background: #1a5c1a; color: white; font-weight: bold; border-radius:5px; }"
            "QPushButton:hover { background: #227722; }"
        )

    def _on_save_profile(self) -> None:
        if self._processor.learner.learned_mask is None:
            self._learn_status.setText("Nothing to save — learn footsteps first.")
            self._learn_status.setStyleSheet(f"color: {RED};")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save footstep profile", "footstep_profile.json",
            "Profile (*.json)")
        if not path:
            return
        try:
            self._processor.learner.save_profile(path)
            self._learn_status.setText(f"Profile saved: {path}")
            self._learn_status.setStyleSheet(f"color: {GREEN};")
        except Exception as exc:
            self._learn_status.setText(f"Save failed: {exc}")
            self._learn_status.setStyleSheet(f"color: {RED};")

    def _on_load_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load footstep profile", "", "Profile (*.json)")
        if not path:
            return
        try:
            learner = self._processor.learner
            learner.load_profile(path)
            self._spectrum_widget.update_spectrum(
                learner.avg_spectrum, learner.learned_mask, learner.peak_freqs)
            peak_str = ",  ".join(f"{f:.0f} Hz" for f in learner.peak_freqs[:10])
            self._peak_label.setText(f"Loaded peaks:  {peak_str}")
            self._use_learned_cb.setEnabled(True)
            self._learn_status.setText(
                f"Profile loaded ({len(learner.peak_freqs)} peaks). "
                "Enable \"Use learned frequencies\" to activate.")
            self._learn_status.setStyleSheet(f"color: {GREEN};")
        except Exception as exc:
            self._learn_status.setText(f"Load failed: {exc}")
            self._learn_status.setStyleSheet(f"color: {RED};")

    def _on_use_learned(self, state: int) -> None:
        enabled = bool(state)
        self._processor.use_learned = enabled
        if enabled:
            self._status_lbl.setText("Running — learned mask active")
            self._status_lbl.setStyleSheet(f"color: {GREEN};")
        elif self._engine and self._engine.is_running():
            self._status_lbl.setText("Running — fixed bandpass mode")
            self._status_lbl.setStyleSheet(f"color: {GREEN};")

    # ------------------------------------------------------------------
    # VU meters
    # ------------------------------------------------------------------

    def _update_vu(self) -> None:
        if self._engine and self._engine.is_running():
            self._vu_in.setValue(int(min(self._engine.vu_in * 300, 100)))
            self._vu_out.setValue(int(min(self._engine.vu_out * 300, 100)))
            if self._processor.footstep_active > 0:
                d = self._processor.last_direction
                arrow = "◀" if d < -0.15 else ("▶" if d > 0.15 else "▲")
                self._step_indicator.setText(f"● STEP {arrow}")
                self._step_indicator.setStyleSheet(
                    f"color: {GREEN}; font-weight: bold; font-size: 14px;")
            else:
                self._step_indicator.setText("● STEP")
                self._step_indicator.setStyleSheet(
                    "color: #333; font-weight: bold; font-size: 14px;")
        else:
            self._vu_in.setValue(0)
            self._vu_out.setValue(0)
            self._step_indicator.setStyleSheet(
                "color: #333; font-weight: bold; font-size: 14px;")

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        p = QPalette()
        p.setColor(QPalette.Window,          QColor(28, 28, 28))
        p.setColor(QPalette.WindowText,      QColor(220, 220, 220))
        p.setColor(QPalette.Base,            QColor(18, 18, 18))
        p.setColor(QPalette.AlternateBase,   QColor(38, 38, 38))
        p.setColor(QPalette.Text,            QColor(220, 220, 220))
        p.setColor(QPalette.Button,          QColor(48, 48, 48))
        p.setColor(QPalette.ButtonText,      QColor(220, 220, 220))
        p.setColor(QPalette.Highlight,       QColor(255, 102, 0))
        p.setColor(QPalette.HighlightedText, Qt.white)
        QApplication.instance().setPalette(p)
        self.setStyleSheet(
            "QGroupBox { border:1px solid #444; border-radius:6px; margin-top:8px; padding-top:8px; }"
            "QGroupBox::title { subcontrol-origin:margin; left:10px; color:#ff6600; }"
            "QComboBox, QLabel { color:#ddd; }"
            "QProgressBar { border:1px solid #555; border-radius:3px; background:#222; }"
            "QPushButton { border-radius:4px; padding:4px 10px; }"
        )

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._engine:
            self._engine.stop()
        event.accept()


# ---------------------------------------------------------------------------

def main() -> None:
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    app.setApplicationName("CoD Footstep Enhancer")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
