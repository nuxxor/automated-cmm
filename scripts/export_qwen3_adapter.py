#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from qwen3_ft_utils import (
    call_with_supported_kwargs,
    load_config,
    load_model_and_tokenizer,
    resolve_path,
)


QWEN_CHATML_TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{- end }}{{ end }}<|im_start|>assistant
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained Qwen3 adapter to merged weights and GGUF.")
    parser.add_argument("--config", default="configs/qwen3_14b_unsloth.json")
    parser.add_argument("--model-path", help="Adapter directory or model path to export.")
    parser.add_argument("--export-dir", help="Override export directory.")
    parser.add_argument("--gguf-quant", default="q4_k_m")
    parser.add_argument("--skip-merged", action="store_true")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--write-modelfile", action="store_true")
    parser.add_argument("--ollama-model", default="automated-cmm-qwen3-14b")
    return parser.parse_args()


def write_modelfile(export_dir: Path, gguf_file: Path, ollama_model: str) -> None:
    modelfile = export_dir / "Modelfile"
    try:
        relative_path = gguf_file.relative_to(export_dir)
    except ValueError:
        relative_path = Path(gguf_file.name)

    content = f"""FROM ./{relative_path.as_posix()}

TEMPLATE \"\"\"{QWEN_CHATML_TEMPLATE}\"\"\"

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Create with:
#   ollama create {ollama_model} -f {modelfile.name}
"""
    modelfile.write_text(content)


def run_checked(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def select_gguf_file(gguf_dir: Path, preferred_quant: str) -> Path | None:
    ggufs = sorted(gguf_dir.glob("*.gguf"))
    if not ggufs:
        return None

    preferred = preferred_quant.lower()
    for path in ggufs:
        if preferred in path.name.lower():
            return path
    return ggufs[0]


def ensure_local_llama_quantize(llama_cpp_dir: Path) -> Path:
    quantize_bin = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if quantize_bin.exists() and convert_script.exists():
        return quantize_bin

    if not llama_cpp_dir.exists():
        raise RuntimeError(
            "GGUF export fallback requires a local llama.cpp checkout at "
            f"{llama_cpp_dir}."
        )
    if shutil.which("cmake") is None:
        raise RuntimeError("GGUF export fallback requires `cmake` on PATH.")

    build_dir = llama_cpp_dir / "build"
    run_checked(
        [
            "cmake",
            "-S",
            str(llama_cpp_dir),
            "-B",
            str(build_dir),
            "-DLLAMA_BUILD_SERVER=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DGGML_CUDA=OFF",
            "-DGGML_NATIVE=OFF",
        ]
    )
    run_checked(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "llama-quantize",
            "-j",
            "8",
        ]
    )
    if not quantize_bin.exists():
        raise RuntimeError(f"Failed to build {quantize_bin}.")
    return quantize_bin


def export_gguf_via_llama_cpp(export_dir: Path, model_name: str, gguf_quant: str) -> Path:
    merged_dir = export_dir / "merged_16bit"
    if not merged_dir.exists():
        raise FileNotFoundError(
            "Merged 16-bit weights are required for manual GGUF export: "
            f"{merged_dir}"
        )

    gguf_dir = export_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    llama_cpp_dir = resolve_path("llama.cpp")
    quantize_bin = ensure_local_llama_quantize(llama_cpp_dir)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

    base_name = model_name.replace("/", "-")
    bf16_file = gguf_dir / f"{base_name}-bf16.gguf"
    quant_file = gguf_dir / f"{base_name}-{gguf_quant.lower()}.gguf"

    if not bf16_file.exists():
        run_checked(
            [
                sys.executable,
                str(convert_script),
                str(merged_dir),
                "--outfile",
                str(bf16_file),
                "--outtype",
                "bf16",
                "--model-name",
                base_name,
                "--use-temp-file",
            ]
        )

    if not quant_file.exists():
        run_checked(
            [
                str(quantize_bin),
                str(bf16_file),
                str(quant_file),
                gguf_quant.upper(),
                "16",
            ]
        )

    return quant_file


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = resolve_path(args.model_path or (resolve_path(config["paths"]["run_dir"]) / "adapter"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    export_dir = resolve_path(args.export_dir or config["paths"]["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    model = None
    tokenizer = None
    if not args.skip_merged or not args.skip_gguf:
        _, _, model, tokenizer = load_model_and_tokenizer(
            str(model_path),
            max_seq_length=config["training"]["max_seq_length"],
            load_in_4bit=True,
            full_finetuning=False,
        )

    if not args.skip_merged:
        merged_dir = export_dir / "merged_16bit"
        merged_dir.mkdir(parents=True, exist_ok=True)
        save_merged = getattr(model, "save_pretrained_merged", None)
        if save_merged is None:
            raise RuntimeError("This Unsloth build does not expose save_pretrained_merged.")
        call_with_supported_kwargs(
            save_merged,
            str(merged_dir),
            tokenizer,
            save_method="merged_16bit",
        )

    gguf_file = None
    if not args.skip_gguf:
        gguf_dir = export_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        save_gguf = getattr(model, "save_pretrained_gguf", None)
        if save_gguf is None:
            raise RuntimeError("This Unsloth build does not expose save_pretrained_gguf.")
        try:
            call_with_supported_kwargs(
                save_gguf,
                str(gguf_dir),
                tokenizer,
                quantization_method=args.gguf_quant,
            )
            gguf_file = select_gguf_file(gguf_dir, args.gguf_quant)
        except Exception:
            gguf_file = export_gguf_via_llama_cpp(
                export_dir,
                args.ollama_model,
                args.gguf_quant,
            )

    if args.write_modelfile:
        if gguf_file is None:
            gguf_dir = export_dir / "gguf"
            gguf_file = select_gguf_file(gguf_dir, args.gguf_quant)
            if gguf_file is None:
                raise FileNotFoundError("No GGUF file found. Export GGUF before writing a Modelfile.")
        write_modelfile(export_dir, gguf_file, args.ollama_model)


if __name__ == "__main__":
    main()
