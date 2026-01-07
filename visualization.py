import torch
import numpy as np
import matplotlib.pyplot as plt
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from config import batch_size, n_dims, n_points, nn_hidden_dim, nn_input_dims, nn_output_dim
from eval import evaluate_model


def plot_predictions_vs_targets(model, n_examples=5, model_type="simple_regression", save_path=None):
    """
    Plot model predictions vs actual targets.
    
    Args:
        model: The transformer model
        n_examples: Number of examples to plot
        model_type: "simple_regression" or "nn"
        save_path: Path to save the figure
    """
    model.eval()
    
    if model_type == "simple_regression":
        xs, ys = generate_linear(n_points, batch_size=1, n_dims=n_dims)
    elif model_type == "nn":
        xs, ys = generate_nn(n_points, batch_size=1, nn_hidden_dim, nn_output_dim, nn_input_dims)
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
        ax = axes[b]
        positions = np.arange(N)
        
        if output_dim == 1:
            ax.plot(positions, tgt_all[b], 'o-', label='Target', alpha=0.7)
            ax.plot(positions, pred_all[b], 's-', label='Prediction', alpha=0.7)
            ax.axvline(N-1, color='r', linestyle='--', alpha=0.5, label='Query')
        else:
            # For multi-dimensional output, plot each dimension
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
        print(f"Predictions plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(metrics, save_path=None):
    """
    Plot distribution of prediction errors.
    
    Args:
        metrics: Dictionary from evaluate_model
        save_path: Path to save the figure
    """
    predictions = metrics['predictions']
    targets = metrics['targets']
    
    if len(predictions.shape) == 2 and predictions.shape[-1] == 1:
        errors = (predictions - targets).squeeze().numpy()
    elif len(predictions.shape) == 2:
        errors = (predictions - targets).numpy().flatten()
    else:
        errors = (predictions - targets).numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(targets.numpy().flatten(), predictions.numpy().flatten(), alpha=0.5)
    min_val = min(targets.min().item(), predictions.min().item())
    max_val = max(targets.max().item(), predictions.max().item())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Target')
    plt.ylabel('Prediction')
    plt.title('Predictions vs Targets Scatter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Error distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_generalization_curves(generalization_results, save_path=None):
    """
    Plot generalization curves for different sequence lengths.
    
    Args:
        generalization_results: Dictionary from evaluate_generalization
        save_path: Path to save the figure
    """
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
        print(f"Generalization curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_few_shot_curves(few_shot_results, save_path=None):
    """
    Plot few-shot learning curves.
    
    Args:
        few_shot_results: Dictionary from evaluate_few_shot_performance
        save_path: Path to save the figure
    """
    k_shots = sorted(few_shot_results.keys())
    mean_losses = [few_shot_results[k]['mean_query_loss'] for k in k_shots]
    std_losses = [few_shot_results[k]['std_query_loss'] for k in k_shots]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(k_shots, mean_losses, yerr=std_losses,
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Number of In-Context Examples (k)')
    plt.ylabel('Query Loss')
    plt.title('Few-Shot Learning Performance')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Few-shot curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_sequence_structure(xs, ys, example_idx=0, max_dims=4, save_path=None):
    """
    Visualize the structure of input sequences.
    
    Args:
        xs: Input sequences (B, N, D)
        ys: Target sequences (B, N, output_dim)
        example_idx: Which example to visualize
        max_dims: Maximum number of dimensions to show
        save_path: Path to save the figure
    """
    B, N, D = xs.shape
    example_idx = min(example_idx, B - 1)
    
    x_example = xs[example_idx].cpu().numpy()  # (N, D)
    y_example = ys[example_idx].cpu().numpy()  # (N, output_dim)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot x values
    ax1 = axes[0]
    positions = np.arange(N)
    for dim in range(min(D, max_dims)):
        ax1.plot(positions, x_example[:, dim], 'o-', label=f'x dim {dim}', alpha=0.7)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('X Value')
    ax1.set_title(f'Input Sequence (Example {example_idx})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot y values
    ax2 = axes[1]
    if len(y_example.shape) == 1:
        ax2.plot(positions, y_example, 's-', label='y', color='green', alpha=0.7)
    else:
        for dim in range(y_example.shape[1]):
            ax2.plot(positions, y_example[:, dim], 's-', 
                    label=f'y dim {dim}', alpha=0.7)
    ax2.axvline(N-1, color='r', linestyle='--', alpha=0.5, label='Query')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Y Value')
    ax2.set_title(f'Target Sequence (Example {example_idx})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Sequence structure plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(losses, save_path=None):
    """
    Plot training loss curves.
    
    Args:
        losses: List or array of loss values
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("Generating test data...")
    xs, ys = generate_linear(n_points, batch_size=1, n_dims=n_dims)
    
    print("Visualizing sequence structure...")
    visualize_sequence_structure(xs, ys, example_idx=0)
    
    print("Plotting predictions vs targets...")
    plot_predictions_vs_targets(model, n_examples=3, model_type="simple_regression")
    
    print("Evaluating and plotting error distribution...")
    metrics = evaluate_model(model, n_test_batches=10, model_type="simple_regression")
    plot_error_distribution(metrics)

