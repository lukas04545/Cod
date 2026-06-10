"""
CoD Footstep Enhancer — GUI entry point.

Requirements:
  pip install sounddevice numpy scipy librosa PyQt5 pyqtgraph

Audio routing (Windows):  Install VB-Cable or Voicemeeter.
  Game  →  VB-Cable Input  →  this app (select VB-Cable Output as input here)
  This app output  →  your headphones / speakers

Audio routing (Linux):  Use PulseAudio/PipeWire virtual sinks.
"""

import sys
import threading
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)
from PyQt5.QtGui import QPalette, QColor

from audio_processor import FootstepEnhancer
from device_manager import input_devices, output_devices, AudioDevice
from stream_engine import StreamEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_slider(min_val: int, max_val: int, default: int, label: str, parent=None):
    """Return (QSlider, QLabel-value) tuple with a descriptor label above."""
    container = QWidget(parent)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    title = QLabel(label)
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)

    slider = QSlider(Qt.Horizontal)
    slider.setMinimum(min_val)
    slider.setMaximum(max_val)
    slider.setValue(default)
    layout.addWidget(slider)

    value_lbl = QLabel(str(default))
    value_lbl.setAlignment(Qt.AlignCenter)
    layout.addWidget(value_lbl)

    slider.valueChanged.connect(lambda v: value_lbl.setText(str(v)))
    return container, slider


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CoD Footstep Enhancer — AI Audio Filter")
        self.setMinimumWidth(680)

        self._processor = FootstepEnhancer()
        self._engine: StreamEngine | None = None

        self._build_ui()
        self._apply_dark_theme()
        self._start_vu_timer()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(12)

        # --- Header ---
        header = QLabel("🎮  CoD Footstep Enhancer")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #ff6600;")
        header.setAlignment(Qt.AlignCenter)
        root.addWidget(header)

        subtitle = QLabel(
            "Amplifies footstep frequencies (150–900 Hz) · Ducks gunshots · Suppresses ambience"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(subtitle)

        # --- Device selection ---
        dev_group = QGroupBox("Audio Devices")
        dev_layout = QVBoxLayout(dev_group)

        in_row = QHBoxLayout()
        in_row.addWidget(QLabel("Input (game audio):"))
        self._in_combo = QComboBox()
        in_row.addWidget(self._in_combo, 1)
        dev_layout.addLayout(in_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output (headphones):"))
        self._out_combo = QComboBox()
        out_row.addWidget(self._out_combo, 1)
        dev_layout.addLayout(out_row)

        self._refresh_btn = QPushButton("↺ Refresh Devices")
        self._refresh_btn.clicked.connect(self._populate_devices)
        dev_layout.addWidget(self._refresh_btn)

        root.addWidget(dev_group)
        self._populate_devices()

        # --- Controls ---
        ctrl_group = QGroupBox("Enhancement Controls")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self._footstep_widget, self._footstep_slider = make_slider(
            10, 100, 40, "Footstep Gain\n(10–100×0.1)"
        )
        ctrl_layout.addWidget(self._footstep_widget)
        self._footstep_slider.valueChanged.connect(self._on_footstep_gain)

        self._ambient_widget, self._ambient_slider = make_slider(
            0, 100, 25, "Ambient Suppress\n(% kept)"
        )
        ctrl_layout.addWidget(self._ambient_widget)
        self._ambient_slider.valueChanged.connect(self._on_ambient_suppress)

        self._duck_widget, self._duck_slider = make_slider(
            0, 50, 10, "Gunshot Duck\n(% kept)"
        )
        ctrl_layout.addWidget(self._duck_widget)
        self._duck_slider.valueChanged.connect(self._on_gunshot_duck)

        root.addWidget(ctrl_group)

        # --- VU meters ---
        vu_group = QGroupBox("Level Meters")
        vu_layout = QVBoxLayout(vu_group)

        in_vu_row = QHBoxLayout()
        in_vu_row.addWidget(QLabel("IN "))
        self._vu_in = QProgressBar()
        self._vu_in.setMaximum(100)
        self._vu_in.setTextVisible(False)
        self._vu_in.setStyleSheet("QProgressBar::chunk { background: #00aaff; }")
        in_vu_row.addWidget(self._vu_in, 1)
        vu_layout.addLayout(in_vu_row)

        out_vu_row = QHBoxLayout()
        out_vu_row.addWidget(QLabel("OUT"))
        self._vu_out = QProgressBar()
        self._vu_out.setMaximum(100)
        self._vu_out.setTextVisible(False)
        self._vu_out.setStyleSheet("QProgressBar::chunk { background: #ff6600; }")
        out_vu_row.addWidget(self._vu_out, 1)
        vu_layout.addLayout(out_vu_row)

        root.addWidget(vu_group)

        # --- Status / toggle ---
        bottom = QHBoxLayout()

        self._enable_cb = QCheckBox("Processing enabled")
        self._enable_cb.setChecked(True)
        self._enable_cb.stateChanged.connect(
            lambda s: setattr(self._processor, "enabled", bool(s))
        )
        bottom.addWidget(self._enable_cb)

        bottom.addStretch()

        self._status_lbl = QLabel("Stopped")
        self._status_lbl.setStyleSheet("color: #888;")
        bottom.addWidget(self._status_lbl)

        self._toggle_btn = QPushButton("▶  Start")
        self._toggle_btn.setFixedHeight(40)
        self._toggle_btn.setStyleSheet(
            "QPushButton { background: #cc4400; color: white; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: #ff6600; }"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        bottom.addWidget(self._toggle_btn)

        root.addLayout(bottom)

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
    # Slider callbacks
    # ------------------------------------------------------------------

    def _on_footstep_gain(self, value: int) -> None:
        self._processor.footstep_gain = value * 0.1

    def _on_ambient_suppress(self, value: int) -> None:
        self._processor.ambient_suppress = value / 100.0

    def _on_gunshot_duck(self, value: int) -> None:
        self._processor.gunshot_duck = value / 100.0

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        if self._engine and self._engine.is_running():
            self._engine.stop()
            self._toggle_btn.setText("▶  Start")
            self._status_lbl.setText("Stopped")
            self._status_lbl.setStyleSheet("color: #888;")
        else:
            in_idx = self._in_combo.currentData()
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
                self._toggle_btn.setText("■  Stop")
                self._status_lbl.setText("Running — listening for footsteps…")
                self._status_lbl.setStyleSheet("color: #00cc66;")
            except Exception as exc:
                self._status_lbl.setText(f"Error: {exc}")
                self._status_lbl.setStyleSheet("color: #ff4444;")

    # ------------------------------------------------------------------
    # VU meter refresh
    # ------------------------------------------------------------------

    def _start_vu_timer(self) -> None:
        self._vu_timer = QTimer(self)
        self._vu_timer.setInterval(50)  # 20 fps
        self._vu_timer.timeout.connect(self._update_vu)
        self._vu_timer.start()

    def _update_vu(self) -> None:
        if self._engine and self._engine.is_running():
            self._vu_in.setValue(int(min(self._engine.vu_in * 300, 100)))
            self._vu_out.setValue(int(min(self._engine.vu_out * 300, 100)))
        else:
            self._vu_in.setValue(0)
            self._vu_out.setValue(0)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(20, 20, 20))
        palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.Highlight, QColor(255, 102, 0))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        QApplication.instance().setPalette(palette)
        self.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #ff6600; }"
            "QComboBox, QLabel { color: #ddd; }"
            "QProgressBar { border: 1px solid #555; border-radius: 3px; background: #222; }"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self._engine:
            self._engine.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("CoD Footstep Enhancer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
