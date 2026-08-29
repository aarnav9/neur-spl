import unittest

import torch
from torch import nn

from cut_to_grow.block_flow import ChannelFlowScores
from cut_to_grow.diagnostics import (
    ChannelAblation,
    ChannelRangeAblation,
    aggregate_channel_scores,
    channel_magnitude_scores,
    collect_gate_sensitivity_scores,
    collect_skip_aware_capacities,
    collect_class_conditioned_capacities,
    binary_auroc,
    paired_binary_metric_difference_ci,
    probe_channel_deletions,
    probe_group_deletions,
    rank_percentile_scores,
    rank_stratified_channels,
    spearman_correlation,
    summarize_probe_correlations,
)
from cut_to_grow.model import ChannelGroup, cifar_resnet18


class DiagnosticsTests(unittest.TestCase):
    def test_channel_ablation_is_temporary(self) -> None:
        model = nn.Sequential(nn.Identity())
        group = ChannelGroup("test", "0", "0", 3)
        inputs = torch.ones(2, 3, 2, 2)
        with ChannelAblation(model, group, 1):
            ablated = model(inputs)
        restored = model(inputs)
        self.assertTrue(bool((ablated[:, 1] == 0).all()))
        self.assertTrue(bool((ablated[:, 0] == 1).all()))
        self.assertTrue(bool((restored == 1).all()))

    def test_channel_range_ablation_is_temporary(self) -> None:
        model = nn.Sequential(nn.Identity())
        group = ChannelGroup("test", "0", "0", 4)
        inputs = torch.ones(2, 4, 2, 2)
        with ChannelRangeAblation(model, group, 1, 3):
            ablated = model(inputs)
        self.assertTrue(bool((ablated[:, 1:3] == 0).all()))
        self.assertTrue(bool((ablated[:, (0, 3)] == 1).all()))
        self.assertTrue(bool((model(inputs) == 1).all()))

    def test_group_magnitude_and_rank_sampling(self) -> None:
        model = cifar_resnet18()
        group = model.channel_groups()[0]
        scores = channel_magnitude_scores(model, group)
        self.assertEqual(tuple(scores.shape), (64,))
        selected = rank_stratified_channels(scores, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected)), 4)
        self.assertTrue(
            torch.equal(
                aggregate_channel_scores(torch.tensor([1, 2, 3, 4, 5]), 2),
                torch.tensor([3, 7, 5]),
            )
        )
        self.assertEqual(rank_percentile_scores([2.0, 1.0, 2.0]), [2 / 3, 1 / 6, 2 / 3])

    def test_tie_aware_spearman_and_layer_macro_summary(self) -> None:
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [30, 20, 10]), -1.0)
        records = [
            {
                "group": "a",
                "magnitude": 1,
                "transport": 3,
                "loss_damage": 1,
                "accuracy_damage": 0.01,
            },
            {
                "group": "a",
                "magnitude": 2,
                "transport": 2,
                "loss_damage": 2,
                "accuracy_damage": 0.02,
            },
            {
                "group": "b",
                "magnitude": 3,
                "transport": 1,
                "loss_damage": 3,
                "accuracy_damage": 0.03,
            },
            {
                "group": "b",
                "magnitude": 4,
                "transport": 0,
                "loss_damage": 4,
                "accuracy_damage": 0.04,
            },
        ]
        summary = summarize_probe_correlations(records)
        self.assertAlmostEqual(summary["magnitude"]["pooled_spearman"], 1.0)
        self.assertAlmostEqual(summary["magnitude"]["layer_macro_spearman"], 1.0)
        self.assertAlmostEqual(summary["transport"]["pooled_spearman"], -1.0)
        self.assertAlmostEqual(
            summary["magnitude"]["top_5_percent_damage_auroc"], 1.0
        )
        self.assertIn(
            "within_layer_shuffle_accuracy_drop_2pt_auroc_pvalue",
            summary["magnitude"],
        )

    def test_paired_binary_metric_difference(self) -> None:
        labels = [False, False, True, True]
        difference, low, high = paired_binary_metric_difference_ci(
            [0.0, 0.1, 0.8, 0.9],
            [0.9, 0.8, 0.1, 0.0],
            labels,
            binary_auroc,
            seed=3,
            replicates=50,
        )
        self.assertEqual(difference, 1.0)
        self.assertEqual(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_collect_gate_sensitivity_scores_aligns_to_groups(self) -> None:
        model = cifar_resnet18(num_classes=3)
        groups = [model.channel_groups()[1], model.channel_groups()[3]]
        loader = [(torch.randn(4, 3, 32, 32), torch.arange(4) % 3)]
        scores = collect_gate_sensitivity_scores(
            model,
            loader,
            groups,
            torch.device("cpu"),
            group_size=16,
            max_batches=1,
        )
        self.assertEqual(set(scores), {group.name for group in groups})
        for group in groups:
            self.assertEqual(scores[group.name].examples, 4)
            self.assertEqual(
                scores[group.name].taylor.numel(), group.channels // 16
            )

    def test_collect_skip_aware_capacities_aligns_residual_gates(self) -> None:
        model = cifar_resnet18(num_classes=3)
        groups = list(model.channel_groups()[1:3])
        loader = [(torch.randn(2, 3, 32, 32), torch.arange(2) % 3)]
        weights, gates = collect_skip_aware_capacities(
            model,
            loader,
            groups,
            torch.device("cpu"),
            max_batches=1,
        )
        self.assertEqual(set(weights), {"layer1.0.conv1", "layer1.0.conv2"})
        self.assertEqual(set(gates), {"layer1.0"})
        self.assertEqual(gates["layer1.0"].examples, 2)
        self.assertEqual(tuple(gates["layer1.0"].skip.shape), (64,))
        self.assertTrue(bool((gates["layer1.0"].skip >= 0).all()))

    def test_collect_class_conditioned_capacities_aligns_commodities(self) -> None:
        model = cifar_resnet18(num_classes=3)
        groups = list(model.channel_groups()[1:3])
        loader = [(torch.randn(3, 3, 32, 32), torch.arange(3))]
        weights, gates = collect_class_conditioned_capacities(
            model,
            loader,
            groups,
            torch.device("cpu"),
            class_to_commodity=(0, 0, 1),
            max_batches=1,
        )
        self.assertEqual(set(weights), {"layer1.0.conv1", "layer1.0.conv2"})
        self.assertEqual(set(gates["layer1.0"].hidden), {0, 1})
        self.assertEqual(gates["layer1.0"].examples, {0: 2, 1: 1})

    def test_deletion_measurements_can_be_reused(self) -> None:
        class TinyClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
                self.bn = nn.BatchNorm2d(2)
                self.fc = nn.Linear(2, 2)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                hidden = self.bn(self.conv(inputs)).mean(dim=(2, 3))
                return self.fc(hidden)

        torch.manual_seed(9)
        model = TinyClassifier()
        loader = [(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)]
        group = ChannelGroup("conv", "conv", "bn", 2)
        transport = {"conv": torch.tensor([0.1, 0.2])}
        first = probe_channel_deletions(
            model,
            loader,
            [group],
            transport,
            None,
            torch.device("cpu"),
            channels_per_score=2,
        )
        second = probe_channel_deletions(
            model,
            loader,
            [group],
            transport,
            None,
            torch.device("cpu"),
            channels_per_score=2,
            cached_records=first,
        )
        self.assertTrue(all(int(record["measurement_reused"]) == 0 for record in first))
        self.assertTrue(all(int(record["measurement_reused"]) == 1 for record in second))
        self.assertEqual(
            [record["ablated_loss"] for record in first],
            [record["ablated_loss"] for record in second],
        )

    def test_group_deletion_measurements_can_be_reused(self) -> None:
        class TinyClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 2, kernel_size=1, bias=False)
                self.bn = nn.BatchNorm2d(2)
                self.fc = nn.Linear(2, 2)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                hidden = self.bn(self.conv(inputs)).mean(dim=(2, 3))
                return self.fc(hidden)

        torch.manual_seed(11)
        model = TinyClassifier()
        loader = [(torch.randn(6, 3, 4, 4), torch.arange(6) % 2)]
        group = ChannelGroup("conv", "conv", "bn", 2)
        transport = {"conv": torch.tensor([0.1, 0.2])}
        flow = {
            "conv": ChannelFlowScores(
                marginal_flow=torch.tensor([0.3, 0.4]),
                group_marginal_flow=torch.tensor([0.3, 0.4]),
                cut_incident=torch.tensor([0.2, 0.5]),
                group_cut_incident=torch.tensor([0.2, 0.5]),
                base_flow=1.0,
                channel_group_size=1,
            )
        }
        first = probe_group_deletions(
            model,
            loader,
            [group],
            transport,
            flow,
            torch.device("cpu"),
            group_size=1,
            groups_per_score=2,
        )
        second = probe_group_deletions(
            model,
            loader,
            [group],
            transport,
            flow,
            torch.device("cpu"),
            group_size=1,
            groups_per_score=2,
            cached_records=first,
        )
        self.assertEqual(len(first), 2)
        self.assertTrue(all(int(record["measurement_reused"]) == 1 for record in second))


if __name__ == "__main__":
    unittest.main()
