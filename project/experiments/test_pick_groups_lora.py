"""Training-mechanics tests on small arrays; no foundation-model weights loaded."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks_p0 import ROOT
from train_pick_groups_lora import select_samples, masked_ce, supervised_logits, batch_from_file


class LoRAAcceptanceTests(unittest.TestCase):
    def test_slice_is_twelve_train_four_validation_without_context_overlap(self):
        selected = select_samples(ROOT / "outputs/pick_groups_sft_20260927_v3")
        self.assertEqual(len(selected["train"]), 12)
        self.assertEqual(len(selected["validation"]), 4)
        self.assertEqual({r["metadata"]["kind"] for r in selected["train"]}, {"diagnosis", "forecast", "update"})
        self.assertEqual({r["scene"] for r in selected["train"]}, {"cover", "blocks"})
        self.assertFalse({r["metadata"]["context_id"] for r in selected["train"]} &
                         {r["metadata"]["context_id"] for r in selected["validation"]})

    def test_loss_normalizes_only_supervised_tokens(self):
        logits = mx.array([[[100., -100.], [0., 0.], [100., -100.]]])
        labels = mx.array([[-100, 1, -100]])
        loss = masked_ce(logits, labels)
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        other = masked_ce(mx.zeros_like(logits), labels)
        self.assertAlmostEqual(loss.item(), other.item(), places=6)

    def test_ignored_tokens_have_zero_logit_gradient(self):
        labels = mx.array([[-100, 1, -100]])
        grad = mx.grad(lambda z: masked_ce(z, labels))(mx.zeros((1, 3, 2)))
        mx.eval(grad)
        np.testing.assert_array_equal(np.asarray(grad[:, 0]), np.zeros((1, 2)))
        np.testing.assert_array_equal(np.asarray(grad[:, 2]), np.zeros((1, 2)))
        self.assertGreater(float(mx.sum(mx.abs(grad[:, 1])).item()), 0.)

    def test_real_forward_interface_uses_causal_default_not_2d_padding_mask(self):
        calls = []
        def model(ids, pixels, mask=None, cache=None, **kwargs):
            calls.append((ids.shape, mask, cache, kwargs["image_grid_thw"].shape))
            return SimpleNamespace(logits=mx.zeros((1, ids.shape[1], 5)))
        batch = {"input_ids": mx.array([[1, 2, 3, 4, 0]]), "pixel_values": mx.zeros((6, 3)),
                 "image_grid_thw": mx.ones((6, 3), dtype=mx.int32)}
        result = supervised_logits(model, batch, 3)
        self.assertEqual(result.shape, (1, 2, 5))
        self.assertEqual(calls, [((1, 4), None, None, (6, 3))])

    def test_bad_label_alignment_and_padding_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="lora-test-") as out:
            root = Path(out)
            path = root / "batch.npz"
            entry = {"batch_file": path.name, "audit": {"target_tokens": 2, "prompt_tokens": 2}}
            values = {"input_ids": np.array([[1, 2, 3, 4]]), "attention_mask": np.array([[1, 1, 1, 1]]),
                      "labels": np.array([[-100, -100, 3, 4]])}
            np.savez(path, **values)
            batch, start = batch_from_file(root, entry)
            self.assertEqual(start, 2)
            values["attention_mask"][0, 0] = 0
            np.savez(path, **values)
            with self.assertRaises(ValueError):
                batch_from_file(root, entry)
            values["attention_mask"][:] = 1
            values["labels"] = np.array([[-100, 2, -100, 4]])
            np.savez(path, **values)
            with self.assertRaises(ValueError):
                batch_from_file(root, entry)


if __name__ == "__main__":
    unittest.main()
