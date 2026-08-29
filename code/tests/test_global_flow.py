import unittest

import torch
from torch import nn

from cut_to_grow.global_flow import (
    GlobalClassGateCapacities,
    model_global_class_flow_scores,
)
from cut_to_grow.model import cifar_resnet18
from cut_to_grow.multicommodity import model_concurrent_flow_scores


class GlobalFlowTests(unittest.TestCase):
    def test_global_graph_scores_selected_residual_groups(self) -> None:
        model = cifar_resnet18()
        weights = {
            name: module.weight.detach().abs() + 1.0e-3
            for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        }
        gates: dict[str, dict[int, torch.Tensor]] = {
            "bn1": {0: torch.ones(64), 1: torch.ones(64)},
            "pool": {0: torch.ones(512), 1: torch.ones(512)},
        }
        for layer_index, layer in enumerate(
            (model.layer1, model.layer2, model.layer3, model.layer4), start=1
        ):
            for block_index, block in enumerate(layer):
                prefix = f"layer{layer_index}.{block_index}"
                for gate in (
                    "bn1",
                    "branch_transport_gate",
                    "skip_transport_gate",
                    "merge_transport_gate",
                ):
                    gates[f"{prefix}.{gate}"] = {
                        0: torch.ones(block.conv2.out_channels),
                        1: torch.ones(block.conv2.out_channels),
                    }
        capacities = GlobalClassGateCapacities(
            by_gate=gates,
            examples={0: 4, 1: 4},
            fine_to_coarse=(0,) * 50 + (1,) * 50,
        )
        scores = model_global_class_flow_scores(
            model,
            weights,
            capacities,
            target_names=("layer1.0.conv1", "layer1.0.conv2"),
            group_size=64,
        )
        self.assertEqual(set(scores), {"layer1.0.conv1", "layer1.0.conv2"})
        self.assertEqual(tuple(scores["layer1.0.conv1"].worst_class_marginal.shape), (1,))
        self.assertEqual(scores["layer1.0.conv2"].commodity_count, 2)

        concurrent = model_concurrent_flow_scores(
            model,
            weights,
            capacities,
            target_names=("layer1.0.conv1", "layer1.0.conv2"),
            group_size=64,
        )
        self.assertGreaterEqual(concurrent["layer1.0.conv1"].concurrent_flow, 0.0)
        self.assertEqual(
            tuple(concurrent["layer1.0.conv2"].dual_deletion_value.shape), (1,)
        )


if __name__ == "__main__":
    unittest.main()
