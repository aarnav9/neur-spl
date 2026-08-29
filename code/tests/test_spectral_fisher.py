import unittest


try:
    import torch
    from torch import nn

    from cut_to_grow.spectral_fisher import (
        GateGradientMatrixMonitor,
        ridge_leverage_scores,
    )
except ImportError:  # pragma: no cover - optional environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SpectralFisherTests(unittest.TestCase):
    def test_monitor_retains_signed_per_example_group_gradients(self) -> None:
        class TinyClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = nn.Identity()
                self.fc = nn.Linear(4, 3, bias=False)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.fc(self.gate(inputs))

        torch.manual_seed(17)
        model = TinyClassifier()
        monitor = GateGradientMatrixMonitor(
            model, module_names=["gate"], group_size=2
        ).attach()
        try:
            inputs = torch.randn(5, 4, requires_grad=True)
            targets = torch.arange(5) % 3
            nn.CrossEntropyLoss()(model(inputs), targets).backward()
        finally:
            monitor.close()
        matrix = monitor.matrix("gate")
        self.assertEqual(tuple(matrix.shape), (5, 2))
        self.assertTrue(bool(torch.isfinite(matrix).all()))
        self.assertTrue(bool((matrix > 0).any()))
        self.assertTrue(bool((matrix < 0).any()))

    def test_ridge_leverage_discounts_duplicate_directions(self) -> None:
        matrix = torch.tensor(
            [
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        fisher, leverage, ridge = ridge_leverage_scores(matrix)
        self.assertGreater(ridge, 0.0)
        self.assertTrue(torch.allclose(fisher, torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)))
        self.assertAlmostEqual(float(leverage[0]), float(leverage[1]))
        self.assertGreater(float(leverage[2]), float(leverage[0]))

    def test_ridge_leverage_handles_zero_operator(self) -> None:
        fisher, leverage, ridge = ridge_leverage_scores(torch.zeros(4, 3))
        self.assertEqual(ridge, 0.0)
        self.assertTrue(bool((fisher == 0).all()))
        self.assertTrue(bool((leverage == 0).all()))


if __name__ == "__main__":
    unittest.main()
