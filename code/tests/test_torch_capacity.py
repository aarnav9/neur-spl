import unittest


try:
    import torch
    from torch import nn

    from cut_to_grow.torch_capacity import GateSensitivityMonitor, TransportCapacityMonitor
except ImportError:  # pragma: no cover - optional environment
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchCapacityTests(unittest.TestCase):
    def test_linear_monitor_records_nonnegative_scores(self) -> None:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(3, 2, bias=False))
        monitor = TransportCapacityMonitor(model, decay=0.5).attach()
        try:
            inputs = torch.tensor(
                [[1.0, -2.0, 0.5], [0.5, 1.0, -1.0]], requires_grad=True
            )
            loss = model(inputs).square().mean()
            loss.backward()
        finally:
            monitor.close()
        self.assertIn("0", monitor.scores)
        self.assertEqual(tuple(monitor.scores["0"].shape), (2, 3))
        self.assertTrue(bool((monitor.scores["0"] >= 0).all()))
        self.assertEqual(tuple(monitor.output_channel_scores("0").shape), (2,))

    def test_monitor_rejects_nonfinite_observations(self) -> None:
        monitor = TransportCapacityMonitor(nn.Linear(1, 1))
        with self.assertRaisesRegex(FloatingPointError, "module 'bad'"):
            monitor._update("bad", torch.tensor([float("nan")]))

    def test_gate_sensitivity_records_structured_scores(self) -> None:
        class TinyClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate = nn.Identity()
                self.fc = nn.Linear(4, 3, bias=False)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                hidden = self.gate(inputs).mean(dim=(2, 3))
                return self.fc(hidden)

        torch.manual_seed(13)
        model = TinyClassifier()
        monitor = GateSensitivityMonitor(
            model, module_names=["gate"], group_size=2
        ).attach()
        try:
            inputs = torch.randn(5, 4, 3, 3, requires_grad=True)
            targets = torch.arange(5) % 3
            nn.CrossEntropyLoss()(model(inputs), targets).backward()
        finally:
            monitor.close()
        scores = monitor.scores("gate")
        self.assertEqual(tuple(scores.taylor.shape), (2,))
        self.assertEqual(tuple(scores.group_fisher.shape), (2,))
        self.assertEqual(scores.examples, 5)
        self.assertTrue(bool(torch.isfinite(scores.taylor).all()))
        self.assertTrue(bool((scores.taylor >= 0).all()))
        self.assertTrue(bool((scores.group_fisher >= 0).all()))
        self.assertTrue(bool((scores.mean_abs_transport >= 0).all()))

    def test_gate_sensitivity_handles_vector_gates(self) -> None:
        model = nn.Sequential(nn.Identity(), nn.Linear(4, 2, bias=False))
        monitor = GateSensitivityMonitor(
            model, module_names=["0"], group_size=2
        ).attach()
        try:
            model(torch.randn(3, 4, requires_grad=True)).square().mean().backward()
        finally:
            monitor.close()
        self.assertEqual(tuple(monitor.scores("0").taylor.shape), (2,))

    def test_gate_sensitivity_collects_conditional_capacities(self) -> None:
        model = nn.Sequential(nn.Linear(3, 3, bias=False))
        monitor = GateSensitivityMonitor(
            model, module_names=["0"], group_size=1
        ).attach()
        try:
            monitor.set_sample_groups(torch.tensor([0, 1, 1]))
            model(torch.randn(3, 3)).square().mean().backward()
        finally:
            monitor.close()
        conditional = monitor.conditional_mean_abs_transport("0")
        self.assertEqual(set(conditional), {0, 1})
        self.assertEqual(tuple(conditional[0].shape), (3,))
        self.assertEqual(monitor.conditional_examples("0"), {0: 1, 1: 2})


if __name__ == "__main__":
    unittest.main()
