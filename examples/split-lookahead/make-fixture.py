#!/usr/bin/env python3
"""Extend the upstream random two-layer Qwen3.5 fixture to four layers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alter-head", action="store_true", help="Make a deliberately different draft")
    parser.add_argument("--swap-head-rows", type=int, nargs=2, help="Change two draft predictions for mixed acceptance")
    parser.add_argument("--bos-id", type=int, help="Override BOS metadata for tokenizer compatibility checks")
    parser.add_argument("--add-bos", choices=("true", "false"), help="Override automatic BOS insertion metadata")
    args = parser.parse_args()
    reader = GGUFReader(args.source)
    if reader.get_field("general.architecture").contents() != "qwen35":
        raise ValueError("expected the upstream qwen35 test fixture")
    if reader.get_field("qwen35.block_count").contents() != 2:
        raise ValueError("expected exactly two source layers")
    writer = GGUFWriter(args.output, "qwen35")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        if field.types[0] == GGUFValueType.ARRAY and not value:
            continue
        if key == "qwen35.block_count":
            value = 4
        elif key.startswith("qwen35.") and field.types[0] == GGUFValueType.ARRAY and len(value) == 2:
            value = value + value
        if key == "tokenizer.ggml.bos_token_id" and args.bos_id is not None:
            continue
        if key == "tokenizer.ggml.add_bos_token" and args.add_bos is not None:
            continue
        writer.add_key_value(key, value, field.types[0], field.types[-1] if len(field.types) > 1 else None)
    if args.bos_id is not None:
        writer.add_bos_token_id(args.bos_id)
    if args.add_bos is not None:
        writer.add_add_bos_token(args.add_bos == "true")
    for tensor in reader.tensors:
        data = np.array(tensor.data, copy=True)
        if args.alter_head and tensor.name == "output.weight":
            data = np.roll(data, 1, axis=0)
        if args.swap_head_rows and tensor.name == "output.weight":
            first, second = args.swap_head_rows
            data[[first, second]] = data[[second, first]]
        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
        if tensor.name.startswith("blk."):
            parts = tensor.name.split(".")
            parts[1] = str(int(parts[1]) + 2)
            writer.add_tensor(".".join(parts), data.copy(), raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
