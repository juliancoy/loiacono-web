#!/usr/bin/env python3
"""Compare the Vulkan Loiacono output with the known-good Python implementation."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from loiacono import Loiacono


SIGNAL_LENGTH = 8192
INPUT_SAMPLES = 1 << 15
FREQUENCY_BANDS = 512
SAMPLE_RATE = 48000.0
MIN_FREQ = 100.0
MAX_FREQ = 12000.0
MULTIPLE = 15

DEFAULT_PLOT_PATH = Path("loiacono_spectrum.png")
DEFAULT_CSV_PATH = Path("loiacono_comparison.csv")


def build_signal() -> np.ndarray:
    """Generate the same test waveform that the Vulkan binary uses."""
    samples = np.zeros(INPUT_SAMPLES, dtype=np.float32)
    amplitude = 0.5
    for i in range(INPUT_SAMPLES):
        t = i / SAMPLE_RATE
        samples[i] = amplitude * (
            np.sin(2 * np.pi * 440.0 * t) + 0.35 * np.sin(2 * np.pi * 880.0 * t)
        )
    return samples


def build_fprime() -> np.ndarray:
    """Return the log-spaced normalized frequencies the GPU uses."""
    low = MIN_FREQ / SAMPLE_RATE
    high = MAX_FREQ / SAMPLE_RATE
    return np.logspace(np.log10(low), np.log10(high), FREQUENCY_BANDS)


def run_python_reference(signal: Sequence[float], fprime: np.ndarray) -> np.ndarray:
    """Compute the spectrum using loiacono.py."""
    inst = Loiacono(fprime=fprime, dtftlen=SIGNAL_LENGTH, multiple=MULTIPLE)
    return inst.run(signal[:SIGNAL_LENGTH])


def parse_gpu_output(stdout: str) -> dict[int, float]:
    """Parse the Vulkan binary's textual spectrum dump."""
    bins = {}
    matcher = re.compile(r"^bin\s+(\d+):\s+([-+0-9.eE]+)")
    for line in stdout.splitlines():
        match = matcher.match(line.strip())
        if match:
            bins[int(match.group(1))] = float(match.group(2))
    return bins


def run_gpu_binary(path: Path) -> str:
    """Invoke the Vulkan binary and return its output."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run `python build.py` so the Vulkan executable exists."
        )
    resolved = path.resolve()
    result = subprocess.run(
        [str(resolved)], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or "Vulkan binary failed without stderr output.")
    return result.stdout


def compare(
    python_spectrum: np.ndarray, gpu_bins: dict[int, float], max_bins: int | None = None
) -> None:
    """Print a per-bin comparison and a small summary."""
    if not gpu_bins:
        raise SystemExit("No spectrum bins were parsed from the Vulkan binary.")

    sorted_bins = sorted(gpu_bins.items())

    if max_bins is None:
        max_bins = sorted_bins[-1][0] + 1

    diffs: list[float] = []
    print("Comparing the first bins reported by the Vulkan binary with Python:")
    for idx, gpu_value in sorted_bins:
        if idx >= len(python_spectrum):
            raise SystemExit(f"Python spectrum has only {len(python_spectrum)} bins.")
        py_value = float(python_spectrum[idx])
        diff = abs(gpu_value - py_value)
        diffs.append(diff)
        rel = diff / (abs(py_value) + 1e-12)
        print(
            f"bin {idx:03}: Vulkan={gpu_value:.6e}  Python={py_value:.6e}  "
            f"abs_diff={diff:.6e}  rel_diff={rel:.6e}"
        )

    diffs_np = np.array(diffs, dtype=np.float32)
    print(
        "Summary: "
        f"max_abs={float(diffs_np.max()):.6e}, "
        f"mean_abs={float(diffs_np.mean()):.6e}"
    )


def build_gpu_spectrum(length: int, gpu_bins: dict[int, float]) -> np.ndarray:
    """Create an array covering all Python bins for plotting."""
    spectrum = np.zeros(length, dtype=np.float32)
    for idx, value in gpu_bins.items():
        if 0 <= idx < length:
            spectrum[idx] = value
    return spectrum


def _can_show_plot() -> bool:
    """Return True when Matplotlib backend supports interactive display."""
    backend = plt.get_backend().lower()
    if backend.startswith("agg"):
        return False
    if os.name != "nt" and "DISPLAY" not in os.environ:
        return False
    return True


def _normalize_for_plot(spectrum: np.ndarray) -> np.ndarray:
    max_val = float(np.max(np.abs(spectrum)))
    if max_val <= 0.0:
        return spectrum
    return spectrum / max_val


def plot_spectra(
    python_spectrum: np.ndarray,
    gpu_spectrum: np.ndarray,
    output_path: Path | None = None,
    show_if_possible: bool = True,
) -> None:
    """Plot the Python and Vulkan spectra on the same axes."""
    plt.figure(figsize=(10, 4))
    plt.plot(
        _normalize_for_plot(python_spectrum), label="Python reference", linewidth=1.5
    )
    plt.plot(
        _normalize_for_plot(gpu_spectrum),
        label="Vulkan binary",
        linewidth=1.5,
        alpha=0.8,
    )
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    plt.title("Loiacono spectrum comparison")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved spectrum plot to {output_path}")

    if show_if_possible and _can_show_plot():
        plt.show()


def export_comparison_csv(
    python_spectrum: np.ndarray, gpu_bins: dict[int, float], path: Path
) -> None:
    """Export spectrum comparisons to CSV for offline analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["bin", "python", "vulkan", "diff", "rel_diff"])
        for idx, py_value in enumerate(python_spectrum):
            gpu_value = float(gpu_bins.get(idx, 0.0))
            diff = py_value - gpu_value
            rel = diff / (abs(py_value) + 1e-12)
            writer.writerow(
                [
                    idx,
                    f"{py_value:.10e}",
                    f"{gpu_value:.10e}",
                    f"{diff:.10e}",
                    f"{rel:.10e}",
                ]
            )
    print(f"Exported comparison CSV to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare loiacono_vulkan.cpp's spectrum with loiacono.py."
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("loiacono_vulkan"),
        help="Path to the Vulkan Loiacono executable.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Save the comparison plot to this file.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Save the bin-wise comparison to this CSV file.",
    )
    args = parser.parse_args()

    signal = build_signal()
    fprime = build_fprime()
    python_spectrum = run_python_reference(signal, fprime)

    stdout = run_gpu_binary(args.binary)
    gpu_bins = parse_gpu_output(stdout)

    compare(python_spectrum, gpu_bins)
    gpu_spectrum = build_gpu_spectrum(len(python_spectrum), gpu_bins)
    plot_spectra(
        python_spectrum,
        gpu_spectrum,
        output_path=args.plot_output,
    )
    export_comparison_csv(python_spectrum, gpu_bins, args.export_csv)


if __name__ == "__main__":
    main()
