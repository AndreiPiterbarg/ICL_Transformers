import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from config import batch_size, n_dims, n_points, nn_hidden_dim, nn_input_dims, nn_output_dim


def plot_predictions_vs_targets(model, n_examples=5, model_type="simple_regression", save_path=None):
    """Plot model predictions vs actual targets."""
    model.eval()
    
    if model_type == "simple_regression":
        xs, ys = generate_linear(n_points, b_size=1, n_dims=n_dims)
    elif model_type == "nn":
        xs, ys = generate_nn(n_points, batch_size=1, nn_hidden_dim = nn_hidden_dim, nn_output_dim = nn_output_dim, nn_input_dims = nn_input_dims)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    with torch.no_grad():
        preds = model(xs, ys)
        B, N, output_dim = ys.shape
        y_pos = torch.arange(1, 2*N, 2, device=preds.device)
        
        if output_dim == 1:
            pred_all = preds[:, y_pos, :].squeeze(-1).cpu().numpy()
            tgt_all = ys.squeeze(-1).cpu().numpy()
        else:
            pred_all = preds[:, y_pos, :].cpu().numpy()
            tgt_all = ys.cpu().numpy()
    
    model.train()
    
    fig, axes = plt.subplots(min(n_examples, B), 1, figsize=(10, 3*min(n_examples, B)))
    if n_examples == 1:
        axes = [axes]
    
    for b in range(min(n_examples, B)):
        ax = axes[b] if n_examples > 1 else axes[0]
        positions = np.arange(N)
        
        if output_dim == 1:
            ax.plot(positions, tgt_all[b], 'o-', label='Target', alpha=0.7)
            ax.plot(positions, pred_all[b], 's-', label='Prediction', alpha=0.7)
            ax.axvline(N-1, color='r', linestyle='--', alpha=0.5, label='Query')
        else:
            for dim in range(output_dim):
                ax.plot(positions, tgt_all[b, :, dim], 'o-', 
                       label=f'Target dim {dim}', alpha=0.7)
                ax.plot(positions, pred_all[b, :, dim], 's-', 
                       label=f'Pred dim {dim}', alpha=0.7)
            ax.axvline(N-1, color='r', linestyle='--', alpha=0.5, label='Query')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title(f'Example {b+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Predictions vs Targets')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_generalization_curves(generalization_results, save_path=None):
    """Plot generalization curves for different sequence lengths."""
    sequence_lengths = sorted(generalization_results.keys())
    mean_losses = [generalization_results[sl]['mean_query_loss'] for sl in sequence_lengths]
    std_losses = [generalization_results[sl]['std_query_loss'] for sl in sequence_lengths]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(sequence_lengths, mean_losses, yerr=std_losses, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Query Loss')
    plt.title('Generalization to Different Sequence Lengths')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    print("Loading model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("Plotting predictions...")
    plot_predictions_vs_targets(model, n_examples=3, model_type="simple_regression")
