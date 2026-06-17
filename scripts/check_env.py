#!/usr/bin/env python3
"""Print runtime environment details for the microscopy preprocessing server."""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return "not installed"
    return str(getattr(module, "__version__", "installed, version unknown"))


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"CWD: {Path.cwd()}")

    for name in ("TMPDIR", "HF_HOME", "TORCH_HOME"):
        value = os.environ.get(name)
        if value:
            print(f"{name}: {value}")

    for module_name in ("numpy", "pandas", "tifffile", "matplotlib", "torch"):
        print(f"{module_name}: {_version(module_name)}")

    try:
        import torch
    except ImportError:
        print("torch CUDA: torch not installed")
        return 0

    cuda_available = torch.cuda.is_available()
    print(f"torch CUDA available: {cuda_available}")
    device_count = torch.cuda.device_count() if cuda_available else 0
    print(f"torch CUDA device count: {device_count}")
    for index in range(device_count):
        print(f"torch CUDA device {index}: {torch.cuda.get_device_name(index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
