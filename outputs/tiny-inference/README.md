# Tiny inference — first working lab

This is the initial code companion to the learning roadmap. It implements a tiny decoder in NumPy with explicit attention and request-local KV caches. It is deliberately small enough to read in one sitting.

**Status:** runnable CPU mechanics prototype with untrained random weights. It does not load GPT-2, produce useful language, batch requests, serve HTTP, or implement paging. Those are milestones in the roadmap. The current byte encoding is not a pretrained model's tokenizer.

## Run

The supplied code was tested with Python 3.9.6 and NumPy 1.26.3 in this workspace. No model download or GPU is required. A current Python 3 environment with a compatible NumPy build is appropriate; `requirements.txt` gives a compatibility range rather than an exact reproduction lock.

```bash
cd outputs/tiny-inference
python3 -m unittest -v test_engine.py
python3 engine.py --prompt 'Hello inference' --tokens 8
python3 engine.py --prompt 'Hello inference' --tokens 8 --no-cache
python3 benchmark.py --prompt-tokens 64 --new-tokens 32 --repeats 5
```

If NumPy is missing in your chosen interpreter, create a project environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Read in this order

1. `Config`: dimensions and capacity.
2. `softmax`, `layer_norm`, and `gelu`: numerical operations.
3. `TinyDecoder.forward`: embedding → attention → MLP → logits.
4. `KVCache`: allocated capacity versus live length.
5. `generate`: prefill once, then append one selected token at a time.
6. `test_engine.py`: the behavior each optimization must preserve.
7. `benchmark.py`: measurement boundary and raw samples.

The uncached path forwards the entire growing prefix each step. The cached path forwards the initial prompt and then only each newly selected token. The causal mask compares absolute query positions with key positions, so cached multi-token chunks cannot attend to their own future tokens.

## Tests and what they establish

The nine tests cover logits across chunk boundaries, causal isolation, greedy cached/uncached equivalence, independent requests, cache memory accounting, capacity errors, invalid input and cross-model cache ownership, stopping/seeded sampling, and stable softmax.

These tests establish internal consistency and important invariants. They are not an independent verification against a pretrained transformer implementation. The next milestone supplies that external reference.

## Benchmark boundaries

`benchmark-local.json` contains an actual run made during preparation: prompt length 32, output length 16, five repeats per mode after warmup. It includes raw samples and medians. It is not a service benchmark or a claim about LLM hardware performance.

The benchmark checks greedy output equivalence, alternates execution order, and measures generator start to first ID, average subsequent ID interval, and total generation time. Timed work includes generation/cache setup and sampling. It excludes model construction, text tokenization, networking, queuing, and display. The last selected token is not forwarded unnecessarily.

The cache preallocates the full configured capacity. `allocated_bytes` reflects that capacity; `used_bytes` reflects currently occupied positions. Both are cache-only values, not peak process memory. Default five-repeat measurements are too small for reliable p99 analysis.

## First hands-on exercises

1. Predict the shapes and annotate them without changing behavior.
2. Sweep prompt lengths 16, 64, and 128 with 16 output tokens. Save separate JSON files and predict the trend before running.
3. Change chunk partitions in the cache test. Explain why an incorrectly offset mask would fail.
4. Temporarily change the mask to allow future tokens, run the causal-isolation test, then restore it. This is an intentional bug exercise.
5. Calculate cache bytes from the model configuration before reading the property.
6. Add one experiment comparing repeated cache concatenation with preallocation. Include allocation costs in the timing boundary.

## Next: pretrained compatibility

Keep the small reference implementation intact and introduce a PyTorch backend. The initial milestone is GPT-2 checkpoint parity with exact tokenizer/configuration and per-layer comparisons. This tiny model is GPT-like but not checkpoint-compatible: it omits learned LayerNorm affine parameters and projection biases, and uses a different vocabulary/configuration.

Suggested eventual modules, to add only as their milestones arrive:

```text
model/        configuration, checkpoint loading, numerical forward pass
runtime/      generation state, sampling, stop rules
cache/        contiguous cache, later block pool and prefix sharing
scheduler/    admission, iteration planning, cancellation
server/       API, streaming, request lifecycle
benchmarks/   workload traces, load driver, raw results
tests/        numerical parity, lifecycle invariants, failures
```

Use a model interface such as `prefill(token_ids, state)` and `decode(next_ids, state)`, but keep request positions and cache ownership explicit. A scheduler decision must eventually execute real batched work; do not confuse looping over requests with continuous batched inference.
