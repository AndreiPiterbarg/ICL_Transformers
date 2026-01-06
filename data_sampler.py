import torch

def generate_linear(n_points, b_size, n_dims):
    xs = torch.randn(b_size, n_points, n_dims)

    w = torch.randn(b_size, n_dims, 1) 
    ys = xs @ w + 0.1 * torch.randn(b_size, n_points, 1)
    return xs, ys

@torch.no_grad()
def generate_nn(n_points, num_batches, hidden_dim, output_dim, input_dim):
    xs = torch.randn(num_batches, n_points, input_dim)
    ys = []

    for i in range(num_batches):
        x = xs[i]
        hidden_layer = torch.nn.Linear(input_dim, hidden_dim)
        output_layer = torch.nn.Linear(hidden_dim, output_dim)


        pass1 = hidden_layer(x)
        activated_pass1 = torch.nn.functional.relu(pass1)
        pass2 = output_layer(activated_pass1)
        ys.append(pass2)
    ys_out = torch.stack(ys)
    return xs, ys_out

def test_linear_generate():
    xs, ys, w =generate_linear(3, 5, 4)
    print(xs.shape)
    print(ys.shape)
    print(w.shape)

def test_nn_generate():
    xs, ys =generate_nn(5, 10, 4, 8, 2)
    print(xs.shape)
    print(ys.shape)



if __name__ == "__main__":
    test_nn_generate()