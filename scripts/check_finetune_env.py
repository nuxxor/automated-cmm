#!/usr/bin/env python3
from __future__ import annotations

import platform
import subprocess
import sys
import importlib


def safe_import(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:
        return f"missing ({exc})"


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")

    try:
        importlib.import_module("unsloth")
    except Exception:
        pass

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        print(f"cuda_version: {getattr(torch.version, 'cuda', 'n/a')}")
        print(f"bf16_supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")
        print(f"gpu_count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"gpu_{idx}: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch: missing ({exc})")

    print(f"unsloth: {safe_import('unsloth')}")
    print(f"transformers: {safe_import('transformers')}")
    print(f"trl: {safe_import('trl')}")
    print(f"peft: {safe_import('peft')}")
    print(f"datasets: {safe_import('datasets')}")
    print(f"bitsandbytes: {safe_import('bitsandbytes')}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("nvidia_smi:")
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")
    except Exception as exc:
        print(f"nvidia_smi: unavailable ({exc})")


if __name__ == "__main__":
    main()
