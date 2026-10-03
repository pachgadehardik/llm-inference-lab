"""Readable CPU decoder: real transformer math, untrained demonstration weights.

Single request, float32, learned positions, pre-LayerNorm, multi-head attention.
This is a mechanics lab, not a trained LLM or a production serving engine.
"""
from dataclasses import dataclass
from typing import Optional
import argparse
import numpy as np


@dataclass(frozen=True)
class Config:
    vocab_size: int = 256
    width: int = 64
    heads: int = 4
    layers: int = 2
    max_context: int = 512

    def __post_init__(self):
        if any(v <= 0 for v in vars(self).values()):
            raise ValueError("All dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def layer_norm(x):
    # Initial affine parameters would be scale=1 and bias=0; omitted in this lab.
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(
        x.var(-1, keepdims=True) + 1e-5
    )


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class KVCache:
    """One request owns one fixed-capacity cache. No globals, no shared mutation."""

    def __init__(self, config, owner):
        shape = (config.layers, config.heads, config.max_context,
                 config.width // config.heads)
        self.k = np.empty(shape, dtype=np.float32)
        self.v = np.empty(shape, dtype=np.float32)
        self.length = 0
        self.owner = owner

    @property
    def allocated_bytes(self):
        return self.k.nbytes + self.v.nbytes

    @property
    def used_bytes(self):
        return self.allocated_bytes * self.length // self.k.shape[2]


class TinyDecoder:
    def __init__(self, config=Config(), seed=7):
        self.config = config
        self._owner = object()
        rng = np.random.default_rng(seed)

        def weight(*shape):
            return (rng.standard_normal(shape) * 0.02).astype(np.float32)

        c = config
        self.embedding = weight(c.vocab_size, c.width)
        self.position = weight(c.max_context, c.width)
        self.blocks = [dict(qkv=weight(c.width, 3 * c.width),
                            out=weight(c.width, c.width),
                            up=weight(c.width, 4 * c.width),
                            down=weight(4 * c.width, c.width))
                       for _ in range(c.layers)]

    def new_cache(self):
        return KVCache(self.config, self._owner)

    def validate_ids(self, token_ids):
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty one-dimensional integer sequence")
        if np.any(ids < 0) or np.any(ids >= self.config.vocab_size):
            raise ValueError("Token ID outside vocabulary")
        return ids

    def forward(self, token_ids, cache: Optional[KVCache] = None):
        """Return [new_tokens, vocabulary] logits; append only new IDs to cache.

        No cache: recompute the whole sequence. Cache: positions begin at its
        current length. A chunk can contain one or multiple new tokens.
        """
        ids = self.validate_ids(token_ids)
        c = self.config
        if cache is not None and cache.owner is not self._owner:
            raise ValueError("Cache belongs to a different model instance")
        start = 0 if cache is None else cache.length
        end = start + len(ids)
        if end > c.max_context:
            raise ValueError("Context capacity exceeded")
        x = self.embedding[ids] + self.position[start:end]
        head_dim = c.width // c.heads
        # Absolute query positions matter when decoding a non-square attention
        # matrix. A fresh triangular mask starting at zero would be incorrect.
        allowed = np.arange(end)[None, :] <= np.arange(start, end)[:, None]

        for layer, block in enumerate(self.blocks):
            q, k, v = np.split(layer_norm(x) @ block["qkv"], 3, axis=-1)
            q, k, v = [a.reshape(len(ids), c.heads, head_dim).transpose(1, 0, 2)
                       for a in (q, k, v)]
            if cache is not None:
                cache.k[layer, :, start:end, :] = k
                cache.v[layer, :, start:end, :] = v
                k = cache.k[layer, :, :end, :]
                v = cache.v[layer, :, :end, :]
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(head_dim)
            probabilities = softmax(np.where(allowed[None, :, :], scores, -np.inf))
            attention = (probabilities @ v).transpose(1, 0, 2).reshape(len(ids), c.width)
            x = x + attention @ block["out"]
            x = x + gelu(layer_norm(x) @ block["up"]) @ block["down"]

        if cache is not None:
            cache.length = end
        # Tied embedding/output weights, as in several real decoder families.
        return layer_norm(x) @ self.embedding.T


def generate(model, prompt_ids, max_new_tokens=16, use_cache=True,
             temperature=0.0, seed=0, eos_id=None):
    """Yield generated IDs. Temperature zero means greedy argmax.

    The last sampled token need not be forwarded because no later prediction
    depends on it. Budget validation is deliberately conservative.
    """
    ids = model.validate_ids(prompt_ids).tolist()
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if eos_id is not None and not 0 <= eos_id < model.config.vocab_size:
        raise ValueError("eos_id outside vocabulary")
    if len(ids) + max_new_tokens > model.config.max_context:
        raise ValueError("Prompt plus output budget exceeds context capacity")
    if max_new_tokens == 0:
        return
    rng = np.random.default_rng(seed)
    cache = model.new_cache() if use_cache else None
    for step in range(max_new_tokens):
        incoming = ids if step == 0 or not use_cache else ids[-1:]
        logits = model.forward(incoming, cache)[-1]
        if temperature == 0:
            token = int(np.argmax(logits))
        else:
            token = int(rng.choice(len(logits), p=softmax(logits / temperature)))
        ids.append(token)
        yield token
        if token == eos_id:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Hello inference")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    model = TinyDecoder()
    # UTF-8 bytes are a reversible input encoding, not GPT-2's BPE tokenizer.
    prompt = list(args.prompt.encode("utf-8"))
    result = list(generate(model, prompt, args.tokens, not args.no_cache))
    print("UNTRAINED mechanics demo; meaningful language is not expected.")
    print("Input byte IDs:", prompt)
    print("Generated IDs:", result)
