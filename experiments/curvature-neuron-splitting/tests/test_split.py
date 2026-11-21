import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurspl.split import split_swiglu_layer  # noqa: E402
from neurspl.trainable_split import (  # noqa: E402
    install_trainable_split,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(5, 7, bias=False)
        self.up_proj = nn.Linear(5, 7, bias=False)
        self.down_proj = nn.Linear(7, 5, bias=False)
        self.act_fn = nn.SiLU()
        self.intermediate_size = 7

    def forward(self, values):
        return self.down_proj(self.act_fn(self.gate_proj(values)) * self.up_proj(values))


class ToyModel(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        layer = nn.Module()
        layer.mlp = mlp
        inner = nn.Module()
        inner.layers = nn.ModuleList([layer])
        self.model = inner

    def forward(self, values):
        return self.model.layers[0].mlp(values)


class SplitTests(unittest.TestCase):
    def test_split_preserves_toy_mlp(self):
        torch.manual_seed(7)
        mlp = ToyMLP()
        model = SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)])
        )
        values = torch.randn(3, 5)
        before = mlp(values)
        record = split_swiglu_layer(model, 0, [1, 5])
        after = mlp(values)
        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))
        self.assertEqual(record.original_width, 7)
        self.assertEqual(record.expanded_width, 9)
        self.assertEqual(record.added_parameters, 30)

    def test_trainable_view_preserves_split_and_parameter_count(self):
        torch.manual_seed(11)
        model = ToyModel(ToyMLP())
        values = torch.randn(4, 5)
        before = model(values)
        parameters_before = sum(parameter.numel() for parameter in model.parameters())

        gate_directions = torch.randn(2, 5)
        up_directions = torch.randn(2, 5)
        adapter, record = install_trainable_split(
            model,
            0,
            [1, 5],
            gate_directions=gate_directions,
            up_directions=up_directions,
            epsilon=0.0,
        )
        after = model(values)
        parameters_after = sum(parameter.numel() for parameter in model.parameters())

        self.assertTrue(torch.allclose(before, after, atol=1e-6, rtol=1e-6))
        self.assertEqual(parameters_after - parameters_before, record.added_parameters)
        self.assertEqual(
            sum(parameter.numel() for parameter in adapter.trainable_parameters()),
            60,
        )
        self.assertTrue(all(not p.requires_grad for p in adapter.base_mlp.parameters()))

    def test_trainable_view_matches_direct_curvature_separation(self):
        torch.manual_seed(17)
        direct = ToyModel(ToyMLP())
        wrapped = ToyModel(ToyMLP())
        wrapped.load_state_dict(direct.state_dict())
        values = torch.randn(2, 3, 5)
        directions_gate = torch.randn(2, 5)
        directions_up = torch.randn(2, 5)
        epsilon = 0.13

        direct_record = split_swiglu_layer(direct, 0, [1, 5])
        direct_mlp = direct.model.layers[0].mlp
        originals = torch.tensor(direct_record.original_indices)
        duplicates = torch.tensor(direct_record.duplicate_indices)
        with torch.no_grad():
            gate_centers = direct_mlp.gate_proj.weight[originals].clone()
            up_centers = direct_mlp.up_proj.weight[originals].clone()
            direct_mlp.gate_proj.weight[originals] = (
                gate_centers + epsilon * directions_gate
            )
            direct_mlp.gate_proj.weight[duplicates] = (
                gate_centers - epsilon * directions_gate
            )
            direct_mlp.up_proj.weight[originals] = (
                up_centers + epsilon * directions_up
            )
            direct_mlp.up_proj.weight[duplicates] = (
                up_centers - epsilon * directions_up
            )

        adapter, _ = install_trainable_split(
            wrapped,
            0,
            [1, 5],
            gate_directions=directions_gate,
            up_directions=directions_up,
            epsilon=epsilon,
        )
        self.assertTrue(
            torch.allclose(direct(values), wrapped(values), atol=1e-6, rtol=1e-6)
        )

        loss = wrapped(values).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in adapter.trainable_parameters()))
        self.assertTrue(all(p.grad is None for p in adapter.base_mlp.parameters()))

    def test_adapter_checkpoint_round_trip(self):
        torch.manual_seed(23)
        source = ToyModel(ToyMLP())
        destination = ToyModel(ToyMLP())
        destination.load_state_dict(source.state_dict())
        values = torch.randn(3, 5)
        directions_gate = torch.randn(2, 5)
        directions_up = torch.randn(2, 5)
        adapter, _ = install_trainable_split(
            source,
            0,
            [1, 5],
            gate_directions=directions_gate,
            up_directions=directions_up,
            epsilon=0.07,
        )
        with torch.no_grad():
            adapter.down_columns.add_(torch.randn_like(adapter.down_columns) * 0.01)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter_checkpoint(
                path,
                adapter,
                {"layer": 0, "selected_neurons": [1, 5], "status": "complete"},
            )
            loaded, record, metadata = load_adapter_checkpoint(destination, path)

        self.assertEqual(record.original_indices, (1, 5))
        self.assertEqual(metadata["status"], "complete")
        self.assertTrue(torch.allclose(source(values), destination(values), atol=1e-6))
        for name, value in adapter.trainable_state().items():
            self.assertTrue(torch.equal(value, loaded.trainable_state()[name]))


if __name__ == "__main__":
    unittest.main()
