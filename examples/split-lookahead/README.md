# Two-host Qwen3.5 rolling lookahead

This opt-in experimental example ports the Hoshino two-stage early-send scheduler to native llama.cpp inference. It supports dense Qwen3.5 without MTP blocks, one request per process, greedy decoding, one draft token, a contiguous target split, and FP16 boundary activations. Both hosts use the same complete, single-file target GGUF, but load only their assigned weights and caches. The suffix owns the output head. The draft uses a full native llama.cpp context on the coordinator. Speculation defaults to off; enable it with `--send-policy early`.

This is a private experimental port, not an upstream submission. It uses a dedicated stage protocol, not the generic ggml RPC backend. Existing llama.cpp commands and default full-model execution remain unchanged. The example is disabled by default.

The same executable also has a `single` mode for the complete native target on one device. It uses the same prepared token IDs, greedy selection and fixed output budget as the split modes, without a draft or network connection. Use this mode for a matched native llama.cpp baseline; its results are separate from historical Python/NF4 measurements.

## Build

From the llama.cpp checkout:

```bash
cmake -S . -B build-split -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_SPLIT_LOOKAHEAD=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-split --target llama-split-lookahead test-split-stage test-split-wire test-llama-archs -j 4
```

For a CUDA build, configure a separate build directory with `-DGGML_CUDA=ON` and the appropriate CUDA architecture/toolkit. CUDA execution and production-sized models have not been qualified for this port. Do not assume CPU qualification establishes graph capture or cross-GPU parity.

## Run on two hosts

The target must be a supported native GGUF, not the Python runtime's bitsandbytes NF4 checkpoint. Give both hosts the same file; its SHA256, vocabulary identity, split, shape and context capacity are checked during handshake. Multi-file GGUF checkpoints are not supported by the example's identity check. The draft may differ in size but must use the exact same token IDs, text, attributes, EOS and EOT. An unused BOS ID may differ only when both tokenizers disable automatic BOS insertion; differing insertion settings are rejected. Target-to-target handshake identity remains strict. Prepared prompt IDs are a comma-separated list or JSON array; tokenization is outside this example.

On B:

```bash
build-split/bin/llama-split-lookahead worker \
  --model /models/target.gguf --split 44 --ctx 8192 \
  --gpu-layers 99 --bind 192.168.31.232:47140
```

On A:

```bash
build-split/bin/llama-split-lookahead coordinator \
  --model /models/target.gguf --draft-model /models/draft.gguf \
  --split 44 --ctx 8192 --gpu-layers 99 \
  --connect 192.168.31.232:47140 \
  --prompt-ids-file /data/input-ids.json --outputs 4096 \
  --send-policy early --prefill-chunk 256 --output result.json
```

Restart the worker for each request. Use `--send-policy off` and omit `--draft-model` for the matching split control. The example requires explicit endpoint addresses. Use trusted interfaces or an authenticated tunnel: CRC32 detects corruption, not hostile peers; the protocol has no authentication or encryption. `--timeout-seconds` defaults to 600 for each socket send/receive operation. TCP_NODELAY is enabled.

## Run the full target on one device

Use the identical GGUF, prepared input IDs, context capacity, prefill chunk and output budget as the two-host run. `single` loads the full target with the native zero/zero stage range; it does not take `--split`, `--draft-model`, `--connect`, `--bind` or `--send-policy`.

```bash
build-split/bin/llama-split-lookahead single \
  --model /models/target.gguf --ctx 69632 --gpu-layers 99 \
  --prompt-ids-file /data/input-ids.json --outputs 4096 \
  --prefill-chunk 256 --output single-result.json
```

The example does not tokenize input or stop at EOS. It emits exactly `--outputs` greedy token IDs. Context capacity must hold all prompt tokens and the first N-1 output tokens fed back for N outputs. Prefill batches are capped by `--prefill-chunk`, including the draft's first catch-up in early split mode; the context batch reservation is capped accordingly instead of scaling with a long `--ctx`.

## Scheduling and state

After prefill, A sends the current confirmed activation. While B computes its prediction, A advances the draft using only confirmed tokens, forks target sequence 0 into private sequence 1, computes the candidate prefix and sends that activation before receiving B's prediction. The candidate identifies its predecessor frame and logical position; it does not claim to know the prediction still being computed.

B checks the candidate against its retained target prediction. A rejection consumes the physical frame ID but does not execute the suffix or advance its logical cache position. A independently checks the predecessor prediction and the rejection acknowledgement, discards its branch, and recomputes the corrected token. On acceptance, A promotes its prepared branch and starts the next candidate while the accepted frame remains in flight. The final output drains without another proposal. For N >= 2 outputs there are N-2 proposals and N-1 suffix decode passes.

Transactions use llama.cpp's sequence copy/removal machinery for both recurrent and attention state. They do not use attention-KV cropping as a substitute for recurrent rollback. Any invalid reply, transport failure or decode error terminates the process and releases its contexts. There is no reconnect or resume protocol.

Prefill uses bounded synchronous chunks. Draft prefill occurs at the first proposal, and subsequent draft calls consume only newly confirmed tokens. There is no draft queue, block speculation, multi-request serving, activation compression beyond FP16, or port of the Python compact-graph implementation. Native llama.cpp backend optimizations remain responsible for execution.

The experimental `llama_model_params.stage_layer_start/end` API selects a prefix or suffix. Zero/zero preserves the full model; `UINT32_MAX` as end selects the model's last layer. Prefix contexts return raw hidden states through the embedding API and no logits. Suffix contexts require hidden-state embedding input. Staged model saving is rejected. Rebuild all consumers because the model-parameter struct changed.

## Reproduce CPU qualification

The upstream fixture generator creates random weights; these tests validate execution and state semantics, not model quality or production performance. Python fixture preparation needs NumPy and uses this checkout's `gguf-py`.

```bash
mkdir -p build-split/fixtures
build-split/bin/test-llama-archs -a '^qwen35$' -s 123 -o build-split/fixtures
python3 examples/split-lookahead/make-fixture.py \
  build-split/fixtures/qwen35-dense.gguf build-split/fixtures/target.gguf
python3 examples/split-lookahead/make-fixture.py \
  build-split/fixtures/qwen35-dense.gguf build-split/fixtures/draft.gguf --alter-head
python3 examples/split-lookahead/make-fixture.py \
  build-split/fixtures/qwen35-dense.gguf build-split/fixtures/mixed-draft.gguf --swap-head-rows 12 13
python3 examples/split-lookahead/make-fixture.py \
  build-split/fixtures/qwen35-dense.gguf build-split/fixtures/bos-draft.gguf --bos-id 11 --add-bos false
python3 examples/split-lookahead/make-fixture.py \
  build-split/fixtures/qwen35-dense.gguf build-split/fixtures/bos-inserting-draft.gguf --bos-id 11 --add-bos true
build-split/bin/test-split-stage build-split/fixtures/target.gguf
ctest --test-dir build-split -R '^test-split-wire$' --output-on-failure
python3 examples/split-lookahead/test-integration.py \
  --binary build-split/bin/llama-split-lookahead \
  --target build-split/fixtures/target.gguf \
  --draft build-split/fixtures/draft.gguf \
  --mixed-draft build-split/fixtures/mixed-draft.gguf \
  --bos-draft build-split/fixtures/bos-draft.gguf \
  --bos-inserting-draft build-split/fixtures/bos-inserting-draft.gguf \
  --output-dir build-split/integration
```

The following local smoke check exercises full-target greedy decoding and verifies that bounded prefill chunking preserves its complete output sequence:

```bash
build-split/bin/llama-split-lookahead single --model build-split/fixtures/target.gguf \
  --ctx 128 --gpu-layers 0 --prompt-ids '[3,17,9,31,6,15]' --outputs 8 \
  --prefill-chunk 1 --output build-split/integration/single-chunk-1.json
build-split/bin/llama-split-lookahead single --model build-split/fixtures/target.gguf \
  --ctx 128 --gpu-layers 0 --prompt-ids '[3,17,9,31,6,15]' --outputs 8 \
  --prefill-chunk 3 --output build-split/integration/single-chunk-3.json
python3 -c 'import json; from pathlib import Path; p=Path("build-split/integration"); a=json.loads((p/"single-chunk-1.json").read_text()); b=json.loads((p/"single-chunk-3.json").read_text()); assert a["tokens"] == b["tokens"] and len(a["tokens"]) == 8'
```

Tests cover three splits of the four-layer hybrid fixture, exact F32 full/staged logits, acceptance/discard and reset, recurrent-only and attention-only stages, FP16 off/early output parity, mixed acceptance, chunked prefill, output budgets of one/two/many, frame fragmentation/truncation/checksum failure, and reply identity/ordering failures. The fixed mixed-draft rows target this seeded fixture and may need updating on another platform if its random generator implementation differs.

Output JSON includes token IDs, proposal/acceptance/rejection counts, suffix decode passes, wire activation frames, TTFT, total time and decode throughput. Request timing starts after model loading/context creation/handshake, before target prefill, and ends at the final received output. It includes request-time draft initialization and graph capture if performed; loading, tokenization, initial context reservation and cleanup are excluded. Decode throughput counts N-1 tokens after the first; it is null for one output. These semantics are not identical to historical Python measurements.

`single` emits `runtime: "llama.cpp-single"`, `mode: "single"`, the complete ordered `tokens` array, `prompt_tokens`, `output_tokens`, `target_decode_calls`, `ttft_ms`, `total_ms` and `decode_tokens_per_second`. It has no split, codec, wire or speculation fields. Its `post-context-prepared-input-to-last-output-v1` timing starts after full-model loading and context creation, immediately before the first prefill batch, and ends after the final greedy output. Decode throughput counts N-1 outputs after the first and is null for N=1. Compare its measured times with split results only after checking the source/model, prepared input, output budget, context capacity, backend settings and run conditions. Cross-mode token IDs may diverge at a greedy decision because the split crosses an FP16 boundary.

Before production use, qualify a CUDA build on both GPUs, compare complete outputs with an identical native split-off control, and measure against the full target on one 5090 D. A different GGUF quantization is a new numerical baseline; historical NF4 token parity or throughput does not transfer to this port.
