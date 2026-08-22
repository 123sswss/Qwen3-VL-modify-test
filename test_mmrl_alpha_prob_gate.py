import unittest

import torch

from MMRLGating import HardConcreteGate


class AlphaProbabilityGateTest(unittest.TestCase):
    def test_probability_is_deterministic_and_controls_residual_gradient(self):
        gate = HardConcreteGate(temperature=0.775)
        logits = torch.tensor([[-2.0], [0.0], [2.0]])
        residual = torch.ones_like(logits, requires_grad=True)

        probability = gate.probability(logits)
        (probability * residual).sum().backward()

        torch.testing.assert_close(probability, torch.sigmoid(logits))
        torch.testing.assert_close(residual.grad, probability)

    def test_eval_path_still_produces_a_binary_gate(self):
        gate = HardConcreteGate(temperature=0.775).eval()
        logits = torch.tensor([[-2.0], [2.0]])

        hard_gate = (gate(logits) > 0.5).to(torch.float32)

        torch.testing.assert_close(hard_gate, torch.tensor([[0.0], [1.0]]))

    def test_batch_mean_gate_removes_sample_identity(self):
        gate = HardConcreteGate(temperature=0.775)
        logits = torch.tensor([[-2.0], [0.0], [2.0]])
        probability = gate.probability(logits)

        mean_gate = gate.batch_mean_probability(logits)

        torch.testing.assert_close(mean_gate.mean(), probability.mean())
        torch.testing.assert_close(mean_gate, mean_gate.mean().expand_as(mean_gate))


if __name__ == "__main__":
    unittest.main()
