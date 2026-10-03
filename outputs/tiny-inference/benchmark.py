"""Warm CPU mechanics benchmark. No tokenizer, network, queue, or model loading.

TTFT here is LOCAL generator-start to first yielded ID, not service TTFT.
Five default repeats are descriptive samples, not reliable tail-latency data.
"""
import argparse
from dataclasses import asdict
import json
import os
import platform
from time import perf_counter
import numpy as np
from engine import Config, TinyDecoder, generate


def run(model, ids, count, cached):
    start = perf_counter()
    times, tokens = [], []
    for token in generate(model, ids, count, use_cache=cached):
        times.append(perf_counter())
        tokens.append(token)
    return tokens, dict(local_ttft_ms=(times[0] - start) * 1000,
                        mean_inter_token_ms=float(np.mean(np.diff(times))) * 1000,
                        total_ms=(times[-1] - start) * 1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.prompt_tokens < 1 or args.new_tokens < 2 or args.repeats < 1:
        parser.error("Need prompt >= 1, new tokens >= 2, repeats >= 1")
    model = TinyDecoder(Config(max_context=args.prompt_tokens + args.new_tokens))
    ids = np.random.default_rng(42).integers(0, 256, args.prompt_tokens).tolist()
    expected, _ = run(model, ids, args.new_tokens, False)
    actual, _ = run(model, ids, args.new_tokens, True)
    assert actual == expected, "Correctness gate failed"
    samples = {"uncached": [], "cached": []}
    for repeat in range(args.repeats):
        # Alternate order to reduce, not eliminate, thermal/order effects.
        for cached in ((False, True) if repeat % 2 == 0 else (True, False)):
            _, metrics = run(model, ids, args.new_tokens, cached)
            samples["cached" if cached else "uncached"].append(metrics)
    cache = model.new_cache()
    model.forward(ids, cache)
    result = dict(
        scope="Untrained tiny CPU model; timings do not predict production LLM performance",
        python=platform.python_version(), numpy=np.__version__,
        platform=platform.platform(), config=asdict(model.config), arguments=vars(args),
        thread_env={k: os.environ.get(k, "unset") for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        cache_allocated_bytes=cache.allocated_bytes,
        cache_used_after_prefill_bytes=cache.used_bytes,
        greedy_outputs_equal=True, raw_samples=samples,
        medians={mode: {k: float(np.median([s[k] for s in runs])) for k in runs[0]}
                 for mode, runs in samples.items()})
    print(json.dumps(result, indent=2))
