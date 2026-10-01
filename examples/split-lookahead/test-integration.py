#!/usr/bin/env python3
"""Exercise actual stage inference and early lookahead over loopback TCP."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path


def run_case(binary: Path, target: Path, draft: Path, output_dir: Path,
             split: int, outputs: int, policy: str, label: str, chunk: int = 256) -> dict:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    endpoint = f"127.0.0.1:{port}"
    common = ["--model", str(target), "--split", str(split), "--ctx", "256",
              "--gpu-layers", "0", "--threads", "2", "--prefill-chunk", str(chunk)]
    worker_log = output_dir / f"{label}-worker.log"
    coordinator_log = output_dir / f"{label}-coordinator.log"
    result = output_dir / f"{label}.json"
    env = dict(os.environ, OMP_NUM_THREADS="2")
    with worker_log.open("w") as log:
        worker = subprocess.Popen([str(binary), "worker", *common, "--bind", endpoint],
                                  stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 30
            while "worker listening" not in worker_log.read_text():
                if worker.poll() is not None:
                    raise RuntimeError(f"worker exited: {worker_log.read_text()[-4000:]}")
                if time.monotonic() > deadline:
                    raise TimeoutError("worker startup timed out")
                time.sleep(0.02)
            command = [str(binary), "coordinator", *common, "--connect", endpoint,
                       "--prompt-ids", "[3,17,9,31,6,15]", "--outputs", str(outputs),
                       "--send-policy", policy, "--output", str(result)]
            if policy == "early":
                command += ["--draft-model", str(draft)]
            with coordinator_log.open("w") as coordinator:
                completed = subprocess.run(command, stdout=coordinator,
                                           stderr=subprocess.STDOUT, timeout=60, env=env)
            if completed.returncode:
                raise RuntimeError(f"coordinator failed: {coordinator_log.read_text()[-4000:]}\nworker: {worker_log.read_text()[-2000:]}")
            if worker.wait(timeout=10):
                raise RuntimeError(f"worker failed: {worker_log.read_text()[-4000:]}")
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
    row = json.loads(result.read_text())
    assert len(row["tokens"]) == outputs, row
    assert row["worker_decode_calls"] == outputs - 1, row
    assert row["proposals"] == (max(0, outputs - 2) if policy == "early" else 0), row
    assert row["accepted"] + row["rejected"] == row["proposals"], row
    assert row["wire_activation_frames"] == (6 + chunk - 1) // chunk + outputs - 1 + row["rejected"], row
    assert row["total_ms"] >= row["ttft_ms"] >= 0, row
    assert (row["decode_tokens_per_second"] is None) == (outputs == 1), row
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mixed-draft", type=Path)
    parser.add_argument("--bos-draft", type=Path, help="Same map, different unused BOS ID, insertion disabled")
    parser.add_argument("--bos-inserting-draft", type=Path, help="Incompatible automatic BOS insertion enabled")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    single_rows = []
    for outputs in (1, 2, 32):
        paired = []
        for chunk in (1, 3):
            label = f"single-n{outputs}-chunk{chunk}"
            result = args.output_dir / f"{label}.json"
            with (args.output_dir / f"{label}.log").open("w") as log:
                subprocess.run([
                    str(args.binary.resolve()), "single", "--model", str(args.target.resolve()),
                    "--ctx", "256", "--gpu-layers", "0", "--threads", "2",
                    "--prefill-chunk", str(chunk), "--prompt-ids", "[3,17,9,31,6,15]",
                    "--outputs", str(outputs), "--output", str(result),
                ], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60,
                   env=dict(os.environ, OMP_NUM_THREADS="2"))
            row = json.loads(result.read_text())
            assert len(row["tokens"]) == outputs and row["prompt_tokens"] == 6, row
            assert row["target_decode_calls"] == outputs - 1, row
            assert row["total_ms"] >= row["ttft_ms"] >= 0, row
            assert (row["decode_tokens_per_second"] is None) == (outputs == 1), row
            paired.append(row)
        assert paired[0]["tokens"] == paired[1]["tokens"], "single chunking changed fixture output"
        single_rows.extend(paired)
    rows = []
    for split, outputs in [(2, 1), (2, 2), (1, 16), (2, 16), (3, 16)]:
        prefix = f"split{split}-n{outputs}"
        baseline = run_case(args.binary.resolve(), args.target.resolve(), args.target.resolve(),
                            args.output_dir, split, outputs, "off", prefix + "-off")
        rows.append(baseline)
        for draft_name, draft in [("same", args.target), ("different", args.draft)]:
            row = run_case(args.binary.resolve(), args.target.resolve(), draft.resolve(),
                           args.output_dir, split, outputs, "early", prefix + "-" + draft_name)
            assert row["tokens"] == baseline["tokens"], f"greedy output mismatch: {prefix}-{draft_name}"
            if outputs > 2:
                assert row["accepted" if draft_name == "same" else "rejected"] > 0, row
            rows.append(row)
    chunked = run_case(args.binary.resolve(), args.target.resolve(), args.target.resolve(),
                       args.output_dir, 2, 16, "off", "chunked-off", chunk=2)
    rows.append(chunked)
    for name, draft in [("same", args.target), ("different", args.draft)]:
        row = run_case(args.binary.resolve(), args.target.resolve(), draft.resolve(),
                       args.output_dir, 2, 16, "early", "chunked-" + name, chunk=2)
        assert row["tokens"] == chunked["tokens"], "chunked prefill output mismatch"
        rows.append(row)
    if args.mixed_draft:
        baseline = run_case(args.binary.resolve(), args.target.resolve(), args.target.resolve(),
                            args.output_dir, 2, 32, "off", "mixed-off")
        row = run_case(args.binary.resolve(), args.target.resolve(), args.mixed_draft.resolve(),
                       args.output_dir, 2, 32, "early", "mixed-early")
        assert row["tokens"] == baseline["tokens"], "mixed acceptance output mismatch"
        assert row["accepted"] > 0 and row["rejected"] > 0, row
        rows += [baseline, row]
    if args.bos_draft:
        baseline = run_case(args.binary.resolve(), args.target.resolve(), args.target.resolve(),
                            args.output_dir, 2, 16, "off", "bos-off")
        row = run_case(args.binary.resolve(), args.target.resolve(), args.bos_draft.resolve(),
                       args.output_dir, 2, 16, "early", "bos-unused-early")
        assert row["tokens"] == baseline["tokens"], "unused BOS metadata changed outputs"
        rows += [baseline, row]
    if args.bos_inserting_draft:
        try:
            run_case(args.binary.resolve(), args.target.resolve(), args.bos_inserting_draft.resolve(),
                     args.output_dir, 2, 16, "early", "bos-insertion-rejected")
        except RuntimeError as error:
            assert "target and draft tokenizer maps differ" in str(error), str(error)
        else:
            raise AssertionError("incompatible BOS insertion metadata was accepted")
    summary = {"status": "passed", "requests": len(rows), "results": rows,
               "single_requests": len(single_rows), "single_results": single_rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(rows)} real-model TCP requests and {len(single_rows)} single requests passed")


if __name__ == "__main__":
    main()
