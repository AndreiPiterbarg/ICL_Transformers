import torch
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import lr, batch_size, n_dims, n_points, nn_hidden_dim, nn_input_dims, nn_output_dim

import wandb

def train_step(model, xs, ys, optimizer):
    optimizer.zero_grad()
    preds = model(xs, ys)                         # (B, 2N, 1)
    B, N, _ = ys.shape

    y_pos = torch.arange(1, 2*N, 2, device=preds.device)

    # targets for ALL y positions (including final query y_N)
    tgt_all = ys.squeeze(-1)                      # (B, N)
    pred_all = preds[:, y_pos, :].squeeze(-1)     # (B, N)

    loss = MSE_nn(pred_all, tgt_all)
    loss.backward()
    #check_gradient_flow(model)
    optimizer.step()
    return loss.detach().item()


def train(model, train_steps=1000, log_every=50, eval_every=None):
    """
    Train the model with optional periodic evaluation.
    
    Args:
        model: The model to train
        train_steps: Number of training steps
        log_every: Log training loss every N steps
        eval_every: Evaluate model every N steps (None to disable)
    """
    from config import eval_every as config_eval_every
    from eval import evaluate_model, print_evaluation_report
    
    if eval_every is None:
        eval_every = config_eval_every
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for i in range(train_steps):
        xs, ys = generate_data(n_points, batch_size, n_dims, model)
        loss = train_step(model, xs, ys, optimizer)
        losses.append(loss)

        if i % log_every == 0:
            print(f"step {i} | query loss: {loss:.6f}")
        
        if eval_every is not None and i > 0 and i % eval_every == 0:
            print(f"\n--- Evaluation at step {i} ---")
            model_type = model.name
            metrics = evaluate_model(model, n_test_batches=5,  make_batch_fn=lambda: generate_linear(n_points, batch_size, n_dims))
            print(f"Test Query Loss: {metrics['mean_query_loss']:.6f} ± {metrics['std_query_loss']:.6f}")
            print()
    
    return losses

def generate_data(n_points, batch_size, n_dims, model):
    if model.name == "simple_regression":
        xs, ys = generate_linear(n_points, batch_size, n_dims)
        return xs, ys
    if model.name == "nn":
        xs, ys = generate_nn(n_points, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims)
        return xs,ys

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def MSE_nn(ys_pred, ys):
    return torch.nn.functional.mse_loss(ys_pred, ys)

def check_gradient_flow(model, tiny=1e-10, huge=1e+2):
    stats = []
    for name, p in model.named_parameters():
        if p.grad is None:
            stats.append((name, None))
            continue
        g = p.grad.detach()
        gnorm = g.norm().item()
        stats.append((name, gnorm))

    # Print or log a compact report
    no_grad = [n for n, g in stats if g is None]
    tiny_grad = [n for n, g in stats if g is not None and g < tiny]
    huge_grad = [n for n, g in stats if g is not None and g > huge]

    print(f"[grad-flow] no_grad: {len(no_grad)} | tiny: {len(tiny_grad)} | huge: {len(huge_grad)}")
    # Uncomment for detail:
    # for n in tiny_grad: print("tiny  ", n)
    # for n in huge_grad: print("huge! ", n)

def visualize_sequence(xs, ys, max_examples=1, max_dims=4):
    """
    Show: x0, y0, x1, y1, ..., x_{N-1}, y_{N-1} (last y zero in inputs).
    """
    import torch
    B, N, D = xs.shape
    model_like_z = TransformerModel._combine(xs, ys)  # uses your combine
    for b in range(min(B, max_examples)):
        print(f"--- Batch {b} ---")
        for t in range(2*N):
            kind = "x" if (t % 2 == 0) else "y"
            vec = model_like_z[b, t, :max_dims].tolist()
            print(f"t={t:02d} [{kind}] : {vec}")
        # Also show the true y for the final slot (to confirm target)
        print("true y_N:", ys[b, -1, 0].item())

def run_gradient_descent():
    m = TransformerModel(n_dims, n_points, name= "simple_regression")
    train(m)

def run_nn_ICL():
    m = NNTransformer(nn_input_dims, n_points, name = "nn")
    train(m)


if __name__ == "__main__":
    run_nn_ICL()