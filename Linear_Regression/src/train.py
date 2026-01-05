import torch
from model import TransformerModel
from data_sampler import generate_linear
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Linear_Regression.src.config import lr, batch_size, n_dims, n_points

import wandb

def train_step(model, xs, ys, optimizer):
    optimizer.zero_grad()

    ys_in = ys.clone()
    ys_in[:, -1, :] = 0.0


    output = model(xs, ys_in, inds=[n_points - 1])  # shape (B, 1)
    target = ys[:, [n_points - 1], 0]               # shape (B, 1)

    loss = mean_squared_error(output, target)
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train(model, train_steps=1000, log_every=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(train_steps):
        # FIXED argument order: (n_points, b_size, n_dims)
        xs, ys, w = generate_linear(n_points, batch_size, n_dims)

        loss = train_step(model, xs, ys, optimizer)

        if i % log_every == 0:
            print(f"step {i} | query loss: {loss:.6f}")


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


if __name__ == "__main__":
    t = TransformerModel(n_dims, n_points)
    train(t)