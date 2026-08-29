import unittest

import torch

from cut_to_grow.block_flow import (
    ResidualGateCapacities,
    ClassConditionalGateCapacities,
    class_conditioned_residual_branch_flow_scores,
    model_residual_flow_scores,
    residual_branch_flow_scores,
    skip_aware_residual_branch_flow_scores,
)
from cut_to_grow.model import cifar_resnet18


class BlockFlowTests(unittest.TestCase):
    def test_exact_branch_node_deletion_damage(self) -> None:
        conv1 = torch.tensor([[[[5.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
        conv2 = torch.tensor([[[[3.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
        result = residual_branch_flow_scores(conv1, conv2, group_size=1)
        self.assertAlmostEqual(result.base_flow, 5.0)
        expected = torch.tensor([0.6, 0.4], dtype=torch.float64)
        self.assertTrue(torch.allclose(result.hidden.marginal_flow, expected))
        self.assertTrue(torch.allclose(result.output.marginal_flow, expected))
        self.assertTrue(torch.allclose(result.hidden.cut_incident, expected))

    def test_coarsened_damage_is_allocated_by_local_transport(self) -> None:
        conv1 = torch.tensor([[[[5.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
        conv2 = torch.tensor([[[[3.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
        result = residual_branch_flow_scores(conv1, conv2, group_size=2)
        self.assertTrue(
            torch.allclose(
                result.hidden.marginal_flow,
                torch.tensor([5.0 / 7.0, 2.0 / 7.0], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(result.hidden.group_marginal_flow, torch.ones(2, dtype=torch.float64))
        )

    def test_measured_skip_can_make_branch_deletion_redundant(self) -> None:
        conv1 = torch.tensor([[[[5.0]]]])
        conv2 = torch.tensor([[[[5.0]]]])
        gates = ResidualGateCapacities(
            branch=torch.tensor([5.0]),
            skip=torch.tensor([5.0]),
            merged=torch.tensor([5.0]),
            examples=8,
        )
        result = skip_aware_residual_branch_flow_scores(
            conv1, conv2, gates, group_size=1
        )
        self.assertAlmostEqual(result.base_flow, 5.0)
        self.assertAlmostEqual(float(result.hidden.marginal_flow[0]), 0.0)
        self.assertAlmostEqual(float(result.output.marginal_flow[0]), 0.0)

    def test_measured_skip_preserves_nonredundant_branch_flow(self) -> None:
        conv1 = torch.tensor([[[[5.0]]]])
        conv2 = torch.tensor([[[[5.0]]]])
        gates = ResidualGateCapacities(
            branch=torch.tensor([5.0]),
            skip=torch.tensor([2.0]),
            merged=torch.tensor([7.0]),
            examples=8,
        )
        result = skip_aware_residual_branch_flow_scores(
            conv1, conv2, gates, group_size=1
        )
        self.assertAlmostEqual(result.base_flow, 7.0)
        expected = torch.tensor([5.0 / 7.0], dtype=torch.float64)
        self.assertTrue(torch.allclose(result.hidden.marginal_flow, expected))
        self.assertTrue(torch.allclose(result.output.marginal_flow, expected))

    def test_class_conditioned_flow_aggregates_commodities(self) -> None:
        conv1 = torch.tensor([[[[5.0]]], [[[2.0]]]])
        conv2 = torch.tensor([[[[5.0]], [[0.0]]], [[[0.0]], [[2.0]]]])
        gates = ClassConditionalGateCapacities(
            hidden={0: torch.tensor([2.0, 0.5]), 1: torch.tensor([0.5, 2.0])},
            output={0: torch.tensor([2.0, 0.5]), 1: torch.tensor([0.5, 2.0])},
            examples={0: 4, 1: 4},
        )
        hidden, output = class_conditioned_residual_branch_flow_scores(
            conv1, conv2, gates, group_size=1
        )
        self.assertEqual(hidden.commodity_count, 2)
        self.assertEqual(tuple(hidden.worst_class_marginal.shape), (2,))
        self.assertEqual(tuple(output.top3_mean_marginal.shape), (2,))
        self.assertTrue(bool((hidden.worst_class_marginal >= hidden.class_mean_marginal).all()))

    def test_model_mapping_covers_every_residual_branch_convolution(self) -> None:
        model = cifar_resnet18()
        modules = dict(model.named_modules())
        weight_scores = {
            group.producer: modules[group.producer].weight.detach().abs()
            for group in model.channel_groups()[1:]
        }
        scores = model_residual_flow_scores(model, weight_scores, group_size=512)
        self.assertEqual(len(scores), 16)
        self.assertEqual(tuple(scores["layer1.0.conv1"].marginal_flow.shape), (64,))
        self.assertEqual(tuple(scores["layer4.1.conv2"].marginal_flow.shape), (512,))


if __name__ == "__main__":
    unittest.main()
