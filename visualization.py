"""
Visualization module for in-context learning experiments.
Based on "What Can Transformers Learn In-Context?" (Garg et al., 2022)
https://arxiv.org/abs/2208.01066

Replicates key figures from the paper:
- Figure 2: In-context learning curves
- Figure 4: OOD robustness
- Figure 5: Performance across function classes
- Figure 6: Capacity analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
import seaborn as sns

from config import batch_size, n_dims, n_points, nn_hidden_dim, nn_input_dims, nn_output_dim

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def get_device(model):
    """Get the device of a model."""
    return next(model.parameters()).device


def plot_training_curves(
    losses: List[float],
    eval_steps: Optional[List[int]] = None,
    eval_metrics: Optional[List] = None,
    save_path: Optional[str] = None,
    window_size: int = 50
):
    """
    Plot training loss curves with optional evaluation metrics.
    
    Args:
        losses: List of training losses
        eval_steps: Steps at which evaluation was performed
        eval_metrics: List of evaluation metrics
        save_path: Path to save figure
        window_size: Window size for smoothing
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    ax = axes[0]
    steps = np.arange(len(losses))
    ax.plot(steps, losses, alpha=0.3, label='Raw', color='blue')
    
    # Smooth
    if len(losses) > window_size:
        smoothed = np.convolve(
            losses,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ax.plot(
            steps[window_size-1:],
            smoothed,
            label=f'Smoothed (window={window_size})',
            color='blue',
            linewidth=2
        )
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Evaluation metrics
    if eval_steps and eval_metrics:
        ax = axes[1]
        
        query_losses = [m.mean_query_loss for m in eval_metrics]
        query_stds = [m.std_query_loss for m in eval_metrics]
        
        ax.errorbar(
            eval_steps,
            query_losses,
            yerr=query_stds,
            marker='o',
            capsize=5,
            label='Query Loss',
            linewidth=2
        )
        
        # Plot baseline if available
        if hasattr(eval_metrics[0], 'baseline_comparisons') and eval_metrics[0].baseline_comparisons:
            baselines = [
                m.baseline_comparisons.get('least_squares_query_loss', 0)
                for m in eval_metrics
            ]
            if any(baselines):
                ax.plot(
                    eval_steps,
                    baselines,
                    '--',
                    label='Least Squares Baseline',
                    linewidth=2,
                    color='red'
                )
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Query Loss')
        ax.set_title('Evaluation Metrics')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        axes[1].text(
            0.5, 0.5,
            'No evaluation data',
            ha='center', va='center',
            transform=axes[1].transAxes
        )
        axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_incontext_learning_curve(
    mean_losses: np.ndarray,
    std_losses: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Plot in-context learning curve showing how loss decreases with examples.
    Replicates Figure 2 from the paper.
    
    Args:
        mean_losses: Mean loss at each position (N,)
        std_losses: Std loss at each position (N,)
        baseline: Optional baseline losses (e.g., least squares)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_examples = np.arange(1, len(mean_losses) + 1)
    
    # Plot model performance
    ax.errorbar(
        n_examples,
        mean_losses,
        yerr=std_losses,
        marker='o',
        capsize=5,
        capthick=2,
        linewidth=2,
        label='Transformer',
        color='blue'
    )
    
    # Plot baseline if provided
    if baseline is not None:
        ax.plot(
            n_examples,
            baseline,
            '--',
            linewidth=2,
            label='Least Squares',
            color='red'
        )
    
    ax.set_xlabel('Number of In-Context Examples', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('In-Context Learning Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved in-context learning curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_generalization_curves(
    generalization_results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot generalization to different sequence lengths.
    
    Args:
        generalization_results: Dict mapping seq_len to metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sequence_lengths = sorted(generalization_results.keys())
    mean_losses = [generalization_results[sl].mean_query_loss for sl in sequence_lengths]
    std_losses = [generalization_results[sl].std_query_loss for sl in sequence_lengths]
    
    # Plot 1: Query loss vs sequence length
    ax = axes[0]
    ax.errorbar(
        sequence_lengths,
        mean_losses,
        yerr=std_losses,
        marker='o',
        capsize=5,
        capthick=2,
        linewidth=2,
        label='Model Query Loss'
    )
    
    # Add baseline if available
    if hasattr(list(generalization_results.values())[0], 'baseline_comparisons'):
        baseline_losses = []
        for sl in sequence_lengths:
            metrics = generalization_results[sl]
            if metrics.baseline_comparisons:
                baseline = metrics.baseline_comparisons.get('least_squares_query_loss', None)
                if baseline:
                    baseline_losses.append(baseline)
                else:
                    baseline_losses.append(np.nan)
            else:
                baseline_losses.append(np.nan)
        
        if not all(np.isnan(baseline_losses)):
            ax.plot(
                sequence_lengths,
                baseline_losses,
                '--',
                linewidth=2,
                label='Least Squares',
                color='red'
            )
    
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Query Loss', fontsize=12)
    ax.set_title('Generalization to Different Sequence Lengths', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Ratio to baseline
    ax = axes[1]
    if hasattr(list(generalization_results.values())[0], 'baseline_comparisons'):
        ratios = []
        for sl in sequence_lengths:
            metrics = generalization_results[sl]
            if metrics.baseline_comparisons:
                baseline = metrics.baseline_comparisons.get('least_squares_query_loss', None)
                if baseline and baseline > 0:
                    ratios.append(metrics.mean_query_loss / baseline)
                else:
                    ratios.append(np.nan)
            else:
                ratios.append(np.nan)
        
        if not all(np.isnan(ratios)):
            ax.plot(
                sequence_lengths,
                ratios,
                marker='s',
                linewidth=2,
                markersize=8,
                color='green'
            )
            ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline', linewidth=2)
            ax.set_xlabel('Sequence Length', fontsize=12)
            ax.set_ylabel('Model Loss / Baseline Loss', fontsize=12)
            ax.set_title('Relative Performance', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No baseline data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'No baseline data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved generalization curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_baseline_comparison(
    metrics,
    save_path: Optional[str] = None
):
    """
    Bar chart comparing model to baseline.
    
    Args:
        metrics: EvaluationMetrics object
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    model_loss = metrics.mean_query_loss
    
    if metrics.baseline_comparisons:
        baseline_loss = metrics.baseline_comparisons.get('least_squares_query_loss', 0)
        
        x = ['Transformer', 'Least Squares']
        y = [model_loss, baseline_loss]
        colors = ['#3498db', '#e74c3c']
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{val:.4f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )
        
        # Add ratio annotation
        if baseline_loss > 0:
            ratio = model_loss / baseline_loss
            ax.text(
                0.5, 0.95,
                f'Ratio: {ratio:.3f}',
                transform=ax.transAxes,
                ha='center',
                fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        ax.set_ylabel('Query Loss (MSE)', fontsize=12)
        ax.set_title('Model vs Baseline Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No baseline data available', ha='center', va='center')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved baseline comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ood_robustness(
    ood_results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot out-of-distribution robustness results.
    Replicates Figure 4 from the paper.
    
    Args:
        ood_results: Dictionary mapping scenario to metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = list(ood_results.keys())
    model_losses = [ood_results[s].mean_query_loss for s in scenarios]
    model_stds = [ood_results[s].std_query_loss for s in scenarios]
    
    # Get baselines
    baseline_losses = []
    for s in scenarios:
        if ood_results[s].baseline_comparisons:
            baseline = ood_results[s].baseline_comparisons.get('least_squares_query_loss', 0)
            baseline_losses.append(baseline)
        else:
            baseline_losses.append(0)
    
    # Clean scenario names
    clean_names = [s.replace('_', ' ').title() for s in scenarios]
    
    # Plot 1: Absolute losses
    ax = axes[0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, model_losses, width, yerr=model_stds,
                   capsize=5, label='Transformer', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, baseline_losses, width,
                   label='Least Squares', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Query Loss', fontsize=12)
    ax.set_title('Out-of-Distribution Robustness', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # Plot 2: Ratios
    ax = axes[1]
    ratios = [m / (b + 1e-10) for m, b in zip(model_losses, baseline_losses)]
    
    colors = ['green' if r < 1.2 else 'orange' if r < 2.0 else 'red' for r in ratios]
    bars = ax.bar(x, ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Model Loss / Baseline Loss', fontsize=12)
    ax.set_title('Relative Performance', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{ratio:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved OOD robustness plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_predictions_vs_targets(
    model,
    n_examples: int = 3,
    model_type: str = "simple_regression",
    save_path: Optional[str] = None
):
    """
    Plot model predictions vs actual targets for visualization.
    
    Args:
        model: Trained model
        n_examples: Number of examples to plot
        model_type: Type of model
        save_path: Path to save figure
    """
    from data_sampler import generate_linear, generate_nn
    
    model.eval()
    device = get_device(model)
    
    if model_type == "simple_regression":
        xs, ys = generate_linear(n_points, n_examples, n_dims)
    elif model_type == "nn":
        xs, ys = generate_nn(n_points, n_examples, nn_hidden_dim, nn_output_dim, nn_input_dims)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Move to device
    xs, ys = xs.to(device), ys.to(device)
    
    with torch.no_grad():
        preds = model(xs, ys)
        B, N, output_dim = ys.shape
        y_pos = torch.arange(1, 2*N, 2, device=device)
        
        pred_all = preds[:, y_pos, :].cpu().numpy()
        tgt_all = ys.cpu().numpy()
    
    model.train()
    
    # Create subplots
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for b in range(n_examples):
        ax = axes[b]
        positions = np.arange(N)
        
        if output_dim == 1:
            # Single output dimension
            ax.plot(positions, tgt_all[b, :, 0], 'o-',
                   label='Target', alpha=0.7, linewidth=2, markersize=8, color='blue')
            ax.plot(positions, pred_all[b, :, 0], 's--',
                   label='Prediction', alpha=0.7, linewidth=2, markersize=8, color='red')
            ax.axvline(N-1, color='green', linestyle=':', alpha=0.5, linewidth=2, label='Query Position')
        else:
            # Multiple output dimensions
            for dim in range(output_dim):
                ax.plot(positions, tgt_all[b, :, dim], 'o-',
                       label=f'Target Dim {dim}', alpha=0.7, linewidth=2, markersize=6)
                ax.plot(positions, pred_all[b, :, dim], 's--',
                       label=f'Pred Dim {dim}', alpha=0.7, linewidth=2, markersize=6)
            ax.axvline(N-1, color='green', linestyle=':', alpha=0.5, linewidth=2, label='Query Position')
        
        ax.set_xlabel('Position in Sequence', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'Example {b+1}: Predictions vs Targets', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model Predictions ({model_type})', fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved predictions plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_dashboard(
    training_history: Dict,
    eval_results: Dict,
    model_type: str,
    save_path: Optional[str] = None
):
    """
    Create a comprehensive summary dashboard with multiple subplots.
    
    Args:
        training_history: Training history dictionary
        eval_results: Evaluation results dictionary
        model_type: Type of model
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Training loss
    ax1 = fig.add_subplot(gs[0, :2])
    losses = training_history.get('losses', [])
    if losses:
        steps = np.arange(len(losses))
        ax1.plot(steps, losses, alpha=0.5, linewidth=1)
        window = min(50, len(losses) // 10)
        if len(losses) > window:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax1.plot(steps[window-1:], smoothed, linewidth=2)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
    
    # 2. Standard evaluation metrics
    ax2 = fig.add_subplot(gs[0, 2])
    if 'standard' in eval_results:
        metrics = eval_results['standard']
        labels = ['Total', 'Query', 'In-Context']
        values = [metrics.mean_total_loss, metrics.mean_query_loss, metrics.mean_incontext_loss]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax2.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Loss')
        ax2.set_title('Evaluation Losses', fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 3. In-context learning curve
    ax3 = fig.add_subplot(gs[1, :2])
    if 'per_position' in eval_results:
        mean_losses = eval_results['per_position']['mean']
        std_losses = eval_results['per_position']['std']
        positions = np.arange(1, len(mean_losses) + 1)
        
        ax3.errorbar(positions, mean_losses, yerr=std_losses,
                    marker='o', capsize=3, linewidth=2)
        ax3.set_xlabel('Number of Examples')
        ax3.set_ylabel('Loss')
        ax3.set_title('In-Context Learning Curve', fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # 4. Generalization
    ax4 = fig.add_subplot(gs[1, 2])
    if 'generalization' in eval_results:
        gen_results = eval_results['generalization']
        seq_lens = sorted(gen_results.keys())
        losses = [gen_results[sl].mean_query_loss for sl in seq_lens]
        
        ax4.plot(seq_lens, losses, marker='o', linewidth=2)
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Query Loss')
        ax4.set_title('Generalization', fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    # 5. OOD robustness (if available)
    ax5 = fig.add_subplot(gs[2, :])
    if 'ood' in eval_results:
        ood_results = eval_results['ood']
        scenarios = list(ood_results.keys())
        model_losses = [ood_results[s].mean_query_loss for s in scenarios]
        
        baseline_losses = []
        for s in scenarios:
            if ood_results[s].baseline_comparisons:
                baseline = ood_results[s].baseline_comparisons.get('least_squares_query_loss', 0)
                baseline_losses.append(baseline)
            else:
                baseline_losses.append(0)
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        clean_names = [s.replace('_', ' ').title() for s in scenarios]
        
        ax5.bar(x - width/2, model_losses, width, label='Model', alpha=0.8)
        ax5.bar(x + width/2, baseline_losses, width, label='Baseline', alpha=0.8)
        
        ax5.set_xlabel('Scenario')
        ax5.set_ylabel('Query Loss')
        ax5.set_title('Out-of-Distribution Robustness', fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(clean_names, rotation=30, ha='right')
        ax5.legend()
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Summary Dashboard: {model_type}', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved summary dashboard to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    """Example usage and testing."""
    from model import TransformerModel
    from data_sampler import generate_linear
    
    print("Testing visualization functions...")
    
    # Create dummy model
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    # Test predictions plot
    print("1. Testing predictions plot...")
    plot_predictions_vs_targets(
        model,
        n_examples=3,
        model_type="simple_regression",
        save_path="test_predictions.png"
    )
    
    # Test in-context learning curve
    print("2. Testing in-context learning curve...")
    mean_losses = np.logspace(-1, -3, n_points)
    std_losses = mean_losses * 0.1
    baseline = mean_losses * 1.2
    
    plot_incontext_learning_curve(
        mean_losses,
        std_losses,
        baseline=baseline,
        save_path="test_incontext.png"
    )
    
    print("\nVisualization tests complete!")
