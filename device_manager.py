"""Utilities for listing and selecting audio devices via sounddevice."""
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioDevice:
    index: int
    name: str
    max_inputs: int
    max_outputs: int
    default_sample_rate: int

    def __str__(self) -> str:
        return f"[{self.index}] {self.name} (in={self.max_inputs}, out={self.max_outputs})"


def list_devices() -> list[AudioDevice]:
    devices = []
    for i, d in enumerate(sd.query_devices()):
        devices.append(
            AudioDevice(
                index=i,
                name=d["name"],
                max_inputs=d["max_input_channels"],
                max_outputs=d["max_output_channels"],
                default_sample_rate=int(d["default_samplerate"]),
            )
        )
    return devices


def input_devices() -> list[AudioDevice]:
    return [d for d in list_devices() if d.max_inputs > 0]


def output_devices() -> list[AudioDevice]:
    return [d for d in list_devices() if d.max_outputs > 0]


def default_input() -> Optional[AudioDevice]:
    idx = sd.default.device[0]
    if idx is None:
        return None
    devices = list_devices()
    return devices[idx] if 0 <= idx < len(devices) else None


def default_output() -> Optional[AudioDevice]:
    idx = sd.default.device[1]
    if idx is None:
        return None
    devices = list_devices()
    return devices[idx] if 0 <= idx < len(devices) else None
