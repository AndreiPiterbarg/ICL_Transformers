import torch
from model import TransformerModel
from config import n_dims, n_points

def test_combine_and_tgt():
    B = 2
    xs = torch.arange(B * n_points * n_dims, dtype=torch.float32).reshape(B, n_points, n_dims)
    ys = torch.arange(B * n_points, dtype=torch.float32).reshape(B, n_points, 1)

    zs = TransformerModel._combine(xs, ys)
    # Check shape: (B, 2*N-1, D)
    assert zs.shape == (B, 2 * n_points - 1, n_dims)

    # Check that the last y is not in zs
    last_y = ys[:, -1, 0]
    # The last y should not appear in zs
    assert not torch.allclose(zs[:, -1, 0], last_y)

    # Simulate tgt extraction as in train_step
    tgt = ys[:, [n_points - 1], 0]
    assert tgt.shape == (B, 1)
    print("All tests passed.")

if __name__ == "__main__":
    test_combine_and_tgt()