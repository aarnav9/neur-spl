import math
import unittest

from cut_to_grow.capacity import (
    ExponentialMovingAverage,
    mean_abs_transport,
    mean_squared_transport,
)


class CapacityTests(unittest.TestCase):
    def test_transport_definitions(self) -> None:
        activations = [1.0, -2.0]
        weights = [0.5, 0.5]
        gradients = [2.0, 1.0]
        self.assertAlmostEqual(mean_abs_transport(activations, weights, gradients), 1.0)
        self.assertAlmostEqual(mean_squared_transport(activations, weights, gradients), 1.0)

    def test_rms_differs_when_energy_is_uneven(self) -> None:
        absolute = mean_abs_transport([1.0, 3.0], [1.0, 1.0], [1.0, 1.0])
        squared = mean_squared_transport([1.0, 3.0], [1.0, 1.0], [1.0, 1.0])
        self.assertEqual(absolute, 2.0)
        self.assertAlmostEqual(squared, math.sqrt(5.0))

    def test_ema(self) -> None:
        ema = ExponentialMovingAverage(decay=0.5)
        self.assertEqual(ema.update(2.0), 2.0)
        self.assertEqual(ema.update(4.0), 3.0)
        self.assertEqual(ema.updates, 2)

    def test_invalid_lengths(self) -> None:
        with self.assertRaises(ValueError):
            mean_abs_transport([1.0], [1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()

