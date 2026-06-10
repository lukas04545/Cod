"""
Manages the live audio stream: captures from selected input device,
processes through FootstepEnhancer, and plays to selected output device.
"""
import threading
import numpy as np
import sounddevice as sd
from audio_processor import FootstepEnhancer


class StreamEngine:
    def __init__(
        self,
        processor: FootstepEnhancer,
        input_device: int | None = None,
        output_device: int | None = None,
        sample_rate: int = 48000,
        block_size: int = 512,
        channels: int = 2,
    ) -> None:
        self.processor = processor
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels

        self._stream: sd.Stream | None = None
        self._lock = threading.Lock()

        # Rolling VU meter values (updated in audio thread, read in GUI thread)
        self.vu_in: float = 0.0
        self.vu_out: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            self._stream = sd.Stream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=(self.input_device, self.output_device),
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
                latency="low",
            )
            self._stream.start()

    def stop(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.vu_in = 0.0
        self.vu_out = 0.0

    def is_running(self) -> bool:
        with self._lock:
            return self._stream is not None and self._stream.active

    def update_devices(
        self,
        input_device: int | None,
        output_device: int | None,
        sample_rate: int,
        channels: int,
    ) -> None:
        """Restart the stream with new device settings."""
        was_running = self.is_running()
        self.stop()
        self.input_device = input_device
        self.output_device = output_device
        self.sample_rate = sample_rate
        self.channels = channels
        # Rebuild rate-dependent processor state for the new sample rate
        self.processor.set_sample_rate(sample_rate)
        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time,
        status: sd.CallbackFlags,
    ) -> None:
        self.vu_in = float(np.sqrt(np.mean(indata ** 2)))
        processed = self.processor.process(indata)
        if processed.ndim == 1:
            processed = np.column_stack([processed] * outdata.shape[1])
        outdata[:] = processed[: outdata.shape[0], : outdata.shape[1]]
        self.vu_out = float(np.sqrt(np.mean(outdata ** 2)))
