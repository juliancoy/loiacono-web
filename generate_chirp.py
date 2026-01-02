#!/usr/bin/env python3
"""Generate the chirp waveform used to test the Loiacono spectrum pipeline."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_chirp(
    duration: float,
    sample_rate: int,
    min_freq: float,
    max_freq: float,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a logarithmic chirp from min_freq to max_freq."""
    samples = int(duration * sample_rate)
    t = np.linspace(0.0, duration, samples, endpoint=False)
    ratio = max_freq / min_freq
    phase = 2.0 * math.pi * min_freq * duration * (
        (ratio ** (t / duration) - 1.0) / math.log(ratio)
    )
    data = amplitude * np.sin(phase)
    return data


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write samples to a mono 16-bit WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = np.clip(samples, -1.0, 1.0)
    scaled = np.int16(scaled * 32767)
    with wave.open(str(path), "w") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(scaled.tobytes())


def plot_spectrogram(
    samples: np.ndarray,
    sample_rate: int,
    output: Path | None = None,
) -> None:
    """Draw a spectrogram for quick inspection."""
    plt.figure(figsize=(10, 4))
    plt.specgram(
        samples,
        NFFT=1024,
        Fs=sample_rate,
        noverlap=512,
        cmap="viridis",
        scale="dB",
        mode="psd",
    )
    plt.title("Chirp spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Spectrogram saved to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Loiacono test chirp.")
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of the chirp in seconds.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=48000,
        help="Sample rate in Hz.",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=100.0,
        help="Starting frequency for the chirp.",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=12000.0,
        help="Ending frequency for the chirp.",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="Amplitude of the chirp (0-1).",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("chirp.wav"),
        help="Path to write the generated WAV file.",
    )
    parser.add_argument(
        "--spectrogram",
        type=Path,
        default=Path("chirp_spectrogram.png"),
        help="Optional PNG path to save the spectrogram.",
    )
    args = parser.parse_args()

    samples = build_chirp(
        duration=args.duration,
        sample_rate=args.rate,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        amplitude=args.amplitude,
    )
    save_wav(args.output_wav, samples, args.rate)
    plot_spectrogram(samples, args.rate, output=args.spectrogram)


if __name__ == "__main__":
    main()
