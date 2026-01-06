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
    output = model(xs, ys_in, inds=[n_points - 1])  # shape (B, 1)
    output_query = output[:, -1, :]

    # target 
    tgt = ys_in[:, -1, :] # will extract the last element of each group of batches



    loss = mean_squared_error(output_query, tgt)
    loss.backward()
    optimizer.step()
    return loss.detach().item()

def train(model, train_steps=1000, log_every=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(train_steps):
    #for i in range(1):


        xs, ys, w = generate_linear(n_points, batch_size, n_dims)

        #print (xs)
        #print (ys)
        loss = train_step(model, xs, ys, optimizer)

        if i % log_every == 0:
            print(f"step {i} | query loss: {loss:.6f}")
        #print(f"step {i} | query loss: {loss:.6f}")

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


if __name__ == "__main__":
    t = TransformerModel(n_dims, n_points)
    train(t)