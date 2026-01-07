import torch
import torch.nn.functional as F
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from config import (
    lr, batch_size, n_dims, n_points, 
    nn_hidden_dim, nn_input_dims, nn_output_dim,
    train_steps, log_every, eval_every
)
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def evaluate_model(model, n_test_batches=10, make_batch_fn=None):
    """
    make_batch_fn should return (xs, ys) with shapes:
      xs: (B, N, D_in), ys: (B, N, 1)
    """
    model.eval()
    total_loss = 0.0
    query_losses = []

    for _ in range(n_test_batches):
        xs, ys = make_batch_fn()
        preds = model(xs, ys)  # (B, 2N, 1)

        B, N, _ = ys.shape
        x_pos = torch.arange(0, 2 * N, 2, device=preds.device)

        pred_at_x = preds.index_select(1, x_pos).squeeze(-1)  # (B, N)
        tgt_y = ys.squeeze(-1)                                # (B, N)

        loss = F.mse_loss(pred_at_x, tgt_y)
        qloss = F.mse_loss(pred_at_x[:, -1], tgt_y[:, -1])

        total_loss += loss.item()
        query_losses.append(qloss.item())

    return {
        "mean_loss": total_loss / n_test_batches,
        "mean_query_loss": float(np.mean(query_losses)),
        "std_query_loss": float(np.std(query_losses)),
    }



def evaluate_generalization(model, sequence_lengths, model_type="simple_regression", n_batches=5):
    """Evaluate model generalization to different sequence lengths."""
    results = {}
    
    for seq_len in sequence_lengths:
        query_losses = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(n_batches):
                if model_type == "simple_regression":
                    xs, ys = generate_linear(seq_len, batch_size, n_dims)
                elif model_type == "nn":
                    xs, ys = generate_nn(seq_len, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                preds = model(xs, ys)
                B, N, output_dim = ys.shape
                x_pos = torch.arange(0, 2*N, 2, device=preds.device)
                
                pred_all = preds[:, x_pos, :]
                tgt_all = ys
                
                if output_dim == 1:
                    pred_all = pred_all.squeeze(-1)
                    tgt_all = tgt_all.squeeze(-1)
                
                query_pred = pred_all[:, -1]
                query_tgt = tgt_all[:, -1]
                query_loss = F.mse_loss(query_pred, query_tgt)
                query_losses.append(query_loss.item())
        
        model.train()
        results[seq_len] = {
            "mean_query_loss": np.mean(query_losses),
            "std_query_loss": np.std(query_losses),
        }
    
    return results


def print_evaluation_report(metrics, title="Evaluation Report"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Mean Loss: {metrics['mean_loss']:.6f}")
    print(f"Mean Query Loss: {metrics['mean_query_loss']:.6f} ± {metrics['std_query_loss']:.6f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("Loading model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("Evaluating model...")
    metrics = evaluate_model(model, n_test_batches=10, model_type="simple_regression")
    print_evaluation_report(metrics, "Standard Evaluation")
    
    print("Testing generalization...")
    gen_results = evaluate_generalization(
        model, 
        sequence_lengths=[5, 10, 15, 20, 25, 30],
        model_type="simple_regression",
        n_batches=5
    )
    print("\nGeneralization Results:")
    for seq_len, result in gen_results.items():
        print(f"  Length {seq_len}: Loss = {result['mean_query_loss']:.6f} ± {result['std_query_loss']:.6f}")
