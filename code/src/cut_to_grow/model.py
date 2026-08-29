"""CIFAR-style ResNet-18 and its output-channel research groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


@dataclass(frozen=True)
class ChannelGroup:
    """A convolution output channel gated at its following BatchNorm output."""

    name: str
    producer: str
    gate: str
    channels: int


class TransportGate(nn.Module):
    """Parameter-free, hookable edge that preserves forward values exactly."""

    def forward(self, inputs: Tensor) -> Tensor:
        # A clone creates a distinct autograd edge. ``nn.Identity`` would expose
        # the total gradient of a tensor shared by the residual and main paths.
        return inputs.clone()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.branch_transport_gate = TransportGate()
        self.skip_transport_gate = TransportGate()
        self.merge_transport_gate = TransportGate()

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs

        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.branch_transport_gate(self.bn2(self.conv2(output)))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        identity = self.skip_transport_gate(identity)
        output = self.merge_transport_gate(output + identity)
        output = self.relu(output)
        return output


class CifarResNet(nn.Module):
    """ResNet-18 adapted to 32x32 inputs (3x3 stem and no max-pool)."""

    def __init__(self, blocks: tuple[int, int, int, int], num_classes: int = 100) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = _conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, blocks[0], stride=1)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, out_channels: int, count: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, count):
            blocks.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.pool(output)
        return self.fc(torch.flatten(output, 1))

    def channel_groups(self) -> tuple[ChannelGroup, ...]:
        """Return stable producer/gate names for the main convolution branches.

        Downsample projections are excluded from the first diagnostic. They are
        skip-path capacity, and mixing them into the main-path result would make
        the intervention harder to interpret.
        """

        groups = [ChannelGroup("stem", "conv1", "bn1", 64)]
        for layer_index, layer in enumerate(
            (self.layer1, self.layer2, self.layer3, self.layer4), start=1
        ):
            for block_index, block in enumerate(layer):
                prefix = f"layer{layer_index}.{block_index}"
                groups.append(
                    ChannelGroup(
                        f"{prefix}.conv1",
                        f"{prefix}.conv1",
                        f"{prefix}.bn1",
                        block.conv1.out_channels,
                    )
                )
                groups.append(
                    ChannelGroup(
                        f"{prefix}.conv2",
                        f"{prefix}.conv2",
                        f"{prefix}.bn2",
                        block.conv2.out_channels,
                    )
                )
        return tuple(groups)


def cifar_resnet18(num_classes: int = 100) -> CifarResNet:
    return CifarResNet((2, 2, 2, 2), num_classes=num_classes)
