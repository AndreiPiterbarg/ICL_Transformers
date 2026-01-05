import torch

def generate_linear(n_points, b_size, n_dims):
    xs = torch.randn(b_size, n_points, n_dims)

    w = torch.randn(b_size, n_dims, 1) 
    ys = xs @ w + 0.1 * torch.randn(b_size, n_points, 1)
    return xs, ys, w


if __name__ == "__main__":
    print(generate_linear(3, 5, 4))