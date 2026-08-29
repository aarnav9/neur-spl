import unittest


try:
    import torch
    from torch import nn

    from cut_to_grow.spectral_pruning import (
        JointChannelMask,
        dependent_round,
        leverage_inclusion_probabilities,
        ridge_relative_operator_error,
        sampled_rescaled_mask,
        topk_mask,
    )
except ImportError:  # pragma: no cover - optional environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SpectralPruningTests(unittest.TestCase):
    def test_topk_mask_has_exact_budget_and_stable_ties(self) -> None:
        mask = topk_mask(torch.tensor([1.0, 3.0, 3.0, 2.0]), 2)
        self.assertEqual(mask.tolist(), [False, True, True, False])

    def test_probabilities_and_rounding_preserve_exact_budget(self) -> None:
        probabilities = leverage_inclusion_probabilities(
            torch.tensor([0.9, 0.4, 0.2, 0.1]), 2
        )
        self.assertAlmostEqual(float(probabilities.sum()), 2.0)
        self.assertTrue(bool((probabilities >= 0).all()))
        self.assertTrue(bool((probabilities <= 1).all()))
        for seed in range(10):
            self.assertEqual(int(dependent_round(probabilities, seed=seed).sum()), 2)

    def test_sampled_rescaling_is_finite_and_exact_budget(self) -> None:
        retained, scales, probabilities = sampled_rescaled_mask(
            torch.tensor([0.8, 0.5, 0.3, 0.1]), keep_count=2, seed=9
        )
        self.assertEqual(int(retained.sum()), 2)
        self.assertTrue(bool(torch.isfinite(scales).all()))
        self.assertTrue(bool((scales[retained] >= 1).all()))
        self.assertTrue(bool((probabilities[retained] > 0).all()))

    def test_joint_mask_is_temporary(self) -> None:
        model = nn.Sequential(nn.Identity())
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        with JointChannelMask(model, {"0": torch.tensor([1.0, 0.0, 2.0])}):
            self.assertTrue(
                torch.equal(model(inputs), torch.tensor([[1.0, 0.0, 6.0]]))
            )
        self.assertTrue(torch.equal(model(inputs), inputs))

    def test_operator_error_is_zero_for_dense_scales(self) -> None:
        matrix = torch.tensor([[1.0, 0.5], [0.0, 1.0], [2.0, -1.0]])
        error = ridge_relative_operator_error(
            matrix, torch.ones(2), ridge=0.1
        )
        self.assertAlmostEqual(error, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
