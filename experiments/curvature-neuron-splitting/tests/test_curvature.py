import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurspl.curvature_proxy import silu_derivatives  # noqa: E402
from neurspl.curvature_refine import build_tridiagonal  # noqa: E402


class CurvatureTests(unittest.TestCase):
    def test_silu_derivatives_match_autograd(self):
        values = torch.tensor([-2.0, -0.3, 0.0, 0.7, 3.0], requires_grad=True)
        first_expected = torch.autograd.grad(
            torch.nn.functional.silu(values).sum(), values, create_graph=True
        )[0]
        second_expected = torch.autograd.grad(first_expected.sum(), values)[0]
        first, second = silu_derivatives(values.detach())
        self.assertTrue(torch.allclose(first, first_expected.detach(), atol=1e-6))
        self.assertTrue(torch.allclose(second, second_expected.detach(), atol=1e-6))

    def test_tridiagonal_layout(self):
        diagonal = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        off = torch.tensor([[0.1, 1.0], [0.2, 2.0]])
        expected = torch.tensor([[1.0, 0.1, 0.0], [0.1, 2.0, 0.2], [0.0, 0.2, 3.0]])
        self.assertTrue(torch.equal(build_tridiagonal(diagonal, off, 0), expected))


if __name__ == "__main__":
    unittest.main()

