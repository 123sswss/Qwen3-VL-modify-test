import unittest

import torch

from utils import attention_pooling


class AttentionPoolingTest(unittest.TestCase):
    def test_segmented_pooling_matches_dense_pooling(self):
        torch.manual_seed(7)
        pooling = attention_pooling(input_dim=6, proj_dim=4).double()
        tokens = torch.randn(8, 6, dtype=torch.double)
        batch_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
        valid_mask = torch.tensor(
            [True, False, True, True, True, False, True, True]
        )

        actual = pooling.forward_vectorized(
            tokens,
            batch_indices,
            batch_size=3,
            valid_mask=valid_mask,
        )
        expected = torch.stack([
            pooling(tokens[(batch_indices == index) & valid_mask])
            for index in range(3)
        ])

        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    def test_empty_segment_uses_layer_norm_zero_state(self):
        pooling = attention_pooling(input_dim=4, proj_dim=3).double()
        with torch.no_grad():
            pooling.ln.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        tokens = torch.randn(2, 4, dtype=torch.double)
        batch_indices = torch.tensor([0, 0])

        actual = pooling.forward_vectorized(tokens, batch_indices, batch_size=2)
        expected_empty = pooling.ln(torch.zeros(4, dtype=torch.double))

        torch.testing.assert_close(actual[1], expected_empty)


if __name__ == "__main__":
    unittest.main()
