"""Capture the same sections and system commands as runtime_environment.txt."""

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def capture_environment():
    def command(*args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            return result.stdout.rstrip("\n")
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"Unavailable: {exc}"

    def package(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    import torch

    sections = {
        "DATE": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "REPOSITORY": "\n".join([
            f"Path: {Path.cwd()}", f"Commit: {command('git', 'rev-parse', 'HEAD')}",
            f"Branch: {command('git', 'branch', '--show-current')}",
            f"Working tree:\n{command('git', 'status', '--short')}",
        ]),
        "HOST": platform.node(),
        "CPU": command("lscpu"),
        "MEMORY": command("free", "-h"),
        "OPERATING SYSTEM": Path("/etc/os-release").read_text().strip(),
        "KERNEL": command("uname", "-a"),
        "STORAGE": command("lsblk", "-d", "-o", "NAME,MODEL,SIZE,ROTA"),
        "PYTHON": f"Python {platform.python_version()}\nExecutable: {sys.executable}",
        "THREAD SETTINGS": "\n".join(
            f"{key}={os.environ.get(key, 'unset')}"
            for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        ),
        "PACKAGE VERSIONS": "\n".join([
            *(f"{label}: {package(name)}" for label, name in [
                ("NumPy", "numpy"), ("pandas", "pandas"), ("OpenCV", "opencv-python-headless"),
                ("LightGBM", "lightgbm"), ("PyTorch", "torch"),
                ("SentenceTransformers", "sentence-transformers"),
            ]),
            f"CUDA available: {torch.cuda.is_available()}",
        ]),
        "GPU UTILITY": command("nvidia-smi") if shutil.which("nvidia-smi") else "nvidia-smi not available",
    }
    return "\n\n".join(f"=== {name} ===\n{value}" for name, value in sections.items()) + "\n"


def environment_command(args):
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Environment file already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    content = capture_environment()
    with output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print(f"Saved runtime environment to {output}")
