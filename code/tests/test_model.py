import unittest

import torch

from cut_to_grow.model import cifar_resnet18


class ModelTests(unittest.TestCase):
    def test_cifar_resnet18_shape_and_groups(self) -> None:
        model = cifar_resnet18(num_classes=100)
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 100))

        groups = model.channel_groups()
        self.assertEqual(len(groups), 17)
        self.assertEqual(groups[0].producer, "conv1")
        self.assertEqual(groups[0].gate, "bn1")
        self.assertEqual(groups[-1].name, "layer4.1.conv2")
        modules = dict(model.named_modules())
        for group in groups:
            self.assertIn(group.producer, modules)
            self.assertIn(group.gate, modules)
        self.assertIn("layer1.0.branch_transport_gate", modules)
        self.assertIn("layer1.0.skip_transport_gate", modules)
        self.assertIn("layer1.0.merge_transport_gate", modules)


if __name__ == "__main__":
    unittest.main()
