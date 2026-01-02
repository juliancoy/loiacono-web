#!/usr/bin/env python3
"""Simple helper to compile the Vulkan Loiacono demo."""

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


def pkgconfig_flags(package: str) -> list[str]:
    try:
        result = subprocess.run(
            ["pkg-config", "--cflags", "--libs", package],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    flags = shlex.split(result.stdout.strip())
    return flags


def compile_shader(shader_src: Path, output: Path) -> None:
    if not shader_src.exists():
        raise SystemExit(f"{shader_src} not found; cannot compile shader.")
    compiler = shutil.which("glslangValidator")
    if compiler is None:
        raise SystemExit("glslangValidator not found; please install it to build the shader.")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [compiler, "-V", str(shader_src), "-o", str(output)]
    print("Compiling shader:", " ".join(shlex.quote(arg) for arg in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Vulkan Loiacono demo.")
    parser.add_argument(
        "--output",
        type=Path,
        default="loiacono_vulkan",
        help="Path where the compiled binary will be written.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("loiacono_vulkan.cpp"),
        help="Source file to compile.",
    )
    parser.add_argument(
        "--shader-src",
        type=Path,
        default=Path("shaders/loiacono.comp"),
        help="GLSL compute shader to compile.",
    )
    parser.add_argument(
        "--shader-output",
        type=Path,
        default=Path("shaders/loiacono.comp.spv"),
        help="Path to write the compiled SPIR-V shader.",
    )
    parser.add_argument(
        "--skip-shader-build",
        action="store_true",
        help="Do not invoke glslangValidator before compiling the binary.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    src_path = (root / args.src).resolve()
    if not src_path.exists():
        raise SystemExit(f"{src_path} not found")

    if not args.skip_shader_build:
        compile_shader((root / args.shader_src).resolve(), (root / args.shader_output).resolve())
        stream_src = (root / "shaders/loiacono_stream.comp").resolve()
        stream_out = (root / "shaders/loiacono_stream.comp.spv").resolve()
        if stream_src.exists():
            compile_shader(stream_src, stream_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    pkg_flags = pkgconfig_flags("vulkan")
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        "-Wall",
        str(src_path),
        "-o",
        str((root / args.output).resolve()),
    ]
    if pkg_flags:
        cmd.extend(pkg_flags)
    else:
        cmd.append("-lvulkan")

    print("Running:", " ".join(shlex.quote(arg) for arg in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
