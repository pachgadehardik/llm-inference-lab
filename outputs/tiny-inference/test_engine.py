import unittest
import numpy as np
from engine import Config, TinyDecoder, generate, softmax


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyDecoder(Config(vocab_size=32, width=16, heads=4,
                                        layers=2, max_context=32))
        self.ids = [1, 7, 3, 9, 2, 4]

    def test_cached_logits_match_full_sequence_across_chunk_sizes(self):
        expected = self.model.forward(self.ids)
        for chunks in ([1, 1, 1, 1, 1, 1], [3, 1, 2], [6], [2, 4]):
            cache = self.model.new_cache()
            outputs = []
            start = 0
            for size in chunks:
                outputs.append(self.model.forward(self.ids[start:start+size], cache))
                start += size
            np.testing.assert_allclose(np.concatenate(outputs), expected,
                                       atol=2e-6, rtol=2e-5)

    def test_future_tokens_cannot_change_past_logits(self):
        before = self.model.forward(self.ids)
        after = self.model.forward(self.ids[:3] + [12, 13, 14])
        np.testing.assert_array_equal(before[:3], after[:3])

    def test_greedy_generation_matches(self):
        for count in (0, 1, 12):
            self.assertEqual(list(generate(self.model, self.ids, count)),
                             list(generate(self.model, self.ids, count, False)))

    def test_interleaved_requests_have_independent_state(self):
        a, b = self.model.new_cache(), self.model.new_cache()
        self.model.forward([1, 2], a)
        self.model.forward([9, 8, 7], b)
        actual = self.model.forward([3], a)[-1]
        np.testing.assert_allclose(actual, self.model.forward([1, 2, 3])[-1],
                                   atol=2e-6, rtol=2e-5)
        self.assertEqual((a.length, b.length), (3, 3))

    def test_cache_memory_accounting(self):
        c = self.model.config
        cache = self.model.new_cache()
        self.model.forward(self.ids, cache)
        self.assertEqual(cache.used_bytes, 2 * c.layers * len(self.ids) * c.width * 4)
        self.assertEqual(cache.allocated_bytes, 2 * c.layers * c.max_context * c.width * 4)

    def test_overflow_rejected_before_cache_length_changes(self):
        cache = self.model.new_cache()
        self.model.forward([1] * 32, cache)
        with self.assertRaises(ValueError):
            self.model.forward([1], cache)
        self.assertEqual(cache.length, 32)

    def test_invalid_input_and_cross_model_cache(self):
        for ids in ([], [-1], [32], [1.2], [[1, 2]]):
            with self.assertRaises(ValueError):
                self.model.forward(ids)
        other = TinyDecoder(self.model.config)
        with self.assertRaises(ValueError):
            other.forward([1], self.model.new_cache())
        with self.assertRaises(ValueError):
            list(generate(self.model, [1], -1))
        with self.assertRaises(ValueError):
            list(generate(self.model, [1], temperature=-1))

    def test_eos_stops_and_sampling_is_seeded(self):
        first = list(generate(self.model, self.ids, 1))[0]
        self.assertEqual(list(generate(self.model, self.ids, 5, eos_id=first)), [first])
        a = list(generate(self.model, self.ids, 5, temperature=0.8, seed=19))
        b = list(generate(self.model, self.ids, 5, temperature=0.8, seed=19))
        self.assertEqual(a, b)

    def test_softmax_is_stable(self):
        actual = softmax(np.array([10000.0, 10000.0]))
        np.testing.assert_allclose(actual, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
