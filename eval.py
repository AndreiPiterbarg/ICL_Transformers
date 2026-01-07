"""
Comprehensive evaluation module for in-context learning transformers.
Based on "What Can Transformers Learn In-Context?" (Garg et al., 2022)
https://arxiv.org/abs/2208.01066
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
from dataclasses import dataclass
from config import batch_size, n_dims, nn_hidden_dim, nn_input_dims, nn_output_dim


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mean_total_loss: float
    std_total_loss: float
    mean_query_loss: float
    std_query_loss: float
    mean_incontext_loss: float
    std_incontext_loss: float
    per_position_losses: Optional[np.ndarray] = None
    baseline_comparisons: Optional[Dict[str, float]] = None


def get_device(model):
    """Get the device of a model."""
    return next(model.parameters()).device


def get_max_sequence_length(model):
    """Get the maximum sequence length the model can handle."""
    if hasattr(model, '_backbone'):
        # GPT2-based model - n_positions is the max
        return model._backbone.config.n_positions // 2  # Divide by 2 because we interleave x and y
    return None


@torch.no_grad()
def evaluate_model(
    model,
    n_test_batches: int = 10,
    make_batch_fn: Optional[Callable] = None,
    compute_baselines: bool = False,
    model_type: str = "simple_regression"
) -> EvaluationMetrics:
    """
    Comprehensive model evaluation with optional baseline comparisons.
    
    Args:
        model: The transformer model to evaluate
        n_test_batches: Number of test batches to evaluate on
        make_batch_fn: Function that generates (xs, ys) batches
        compute_baselines: Whether to compute baseline comparisons
        model_type: Type of model ("simple_regression" or "nn")
    
    Returns:
        EvaluationMetrics object containing all evaluation results
    """
    model.eval()
    device = get_device(model)
    
    total_losses = []
    query_losses = []
    incontext_losses = []
    per_position_losses_list = []
    
    baseline_query_losses = [] if compute_baselines else None
    
    for _ in range(n_test_batches):
        xs, ys = make_batch_fn()
        xs, ys = xs.to(device), ys.to(device)
        
        preds = model(xs, ys)
        
        B, N, output_dim = ys.shape
        
        # Get predictions at y positions: 1, 3, 5, ..., 2N-1
        y_pos = torch.arange(1, 2 * N, 2, device=device)
        pred_at_y = preds[:, y_pos, :]  # (B, N, output_dim)
        
        # Compute losses
        total_loss = F.mse_loss(pred_at_y, ys)
        query_loss = F.mse_loss(pred_at_y[:, -1, :], ys[:, -1, :])
        
        # In-context loss (all positions except query)
        if N > 1:
            incontext_loss = F.mse_loss(pred_at_y[:, :-1, :], ys[:, :-1, :])
            incontext_losses.append(incontext_loss.item())
        
        total_losses.append(total_loss.item())
        query_losses.append(query_loss.item())
        
        # Per-position losses
        pos_losses = []
        for i in range(N):
            pos_loss = F.mse_loss(pred_at_y[:, i, :], ys[:, i, :])
            pos_losses.append(pos_loss.item())
        per_position_losses_list.append(pos_losses)
        
        # Compute baseline if requested
        if compute_baselines and model_type == "simple_regression":
            baseline_loss = compute_least_squares_baseline(xs, ys)
            baseline_query_losses.append(baseline_loss)
    
    model.train()
    
    # Aggregate metrics
    metrics = EvaluationMetrics(
        mean_total_loss=float(np.mean(total_losses)),
        std_total_loss=float(np.std(total_losses)),
        mean_query_loss=float(np.mean(query_losses)),
        std_query_loss=float(np.std(query_losses)),
        mean_incontext_loss=float(np.mean(incontext_losses)) if incontext_losses else 0.0,
        std_incontext_loss=float(np.std(incontext_losses)) if incontext_losses else 0.0,
        per_position_losses=np.mean(per_position_losses_list, axis=0)
    )
    
    if compute_baselines:
        metrics.baseline_comparisons = {
            'least_squares_query_loss': float(np.mean(baseline_query_losses))
        }
    
    return metrics


@torch.no_grad()
def compute_least_squares_baseline(xs: torch.Tensor, ys: torch.Tensor) -> float:
    """
    Compute optimal least squares baseline for linear regression.
    
    Args:
        xs: Input features (B, N, D)
        ys: Target values (B, N, 1)
    
    Returns:
        Mean squared error of least squares predictions on query point
    """
    B, N, D = xs.shape
    device = xs.device
    
    # Use N-1 context examples to predict the Nth
    xs_context = xs[:, :-1, :]  # (B, N-1, D)
    ys_context = ys[:, :-1, :]  # (B, N-1, 1)
    xs_query = xs[:, -1:, :]    # (B, 1, D)
    ys_query = ys[:, -1:, :]    # (B, 1, 1)
    
    query_losses = []
    
    for b in range(B):
        X = xs_context[b]  # (N-1, D)
        Y = ys_context[b].squeeze(-1)  # (N-1,)
        
        # Solve least squares: w = (X^T X)^{-1} X^T Y
        try:
            XtX = X.T @ X
            # Add small regularization for numerical stability
            XtX_reg = XtX + 1e-5 * torch.eye(D, device=device)
            w_ls = torch.linalg.solve(XtX_reg, X.T @ Y)
            
            # Predict on query
            y_pred = xs_query[b] @ w_ls.unsqueeze(-1)
            loss = F.mse_loss(y_pred, ys_query[b])
            query_losses.append(loss.item())
        except:
            # If solve fails, use high loss
            query_losses.append(float('inf'))
    
    return float(np.mean(query_losses))


@torch.no_grad()
def evaluate_sequence_length_generalization(
    model,
    sequence_lengths: List[int],
    model_type: str = "simple_regression",
    n_batches: int = 10
) -> Dict[int, EvaluationMetrics]:
    """
    Evaluate model performance across different sequence lengths.
    Tests generalization beyond training length (Figure 2 in paper).
    
    Args:
        model: The transformer model
        sequence_lengths: List of sequence lengths to test
        model_type: Type of model
        n_batches: Number of batches per length
    
    Returns:
        Dictionary mapping sequence length to evaluation metrics
    """
    from data_sampler import generate_linear, generate_nn
    
    # Get the maximum sequence length the model can handle
    max_seq_len = get_max_sequence_length(model)
    
    results = {}
    model.eval()
    device = get_device(model)
    
    for seq_len in sequence_lengths:
        # Skip if sequence length exceeds model capacity
        if max_seq_len is not None and seq_len > max_seq_len:
            print(f"Skipping seq_len={seq_len} (exceeds model max of {max_seq_len})")
            continue
        
        # Create data generator for this sequence length
        if model_type == "simple_regression":
            make_batch_fn = lambda sl=seq_len: generate_linear(sl, batch_size, n_dims)
        elif model_type == "nn":
            make_batch_fn = lambda sl=seq_len: generate_nn(
                sl, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Evaluate at this length
        try:
            metrics = evaluate_model(
                model,
                n_test_batches=n_batches,
                make_batch_fn=make_batch_fn,
                compute_baselines=(model_type == "simple_regression"),
                model_type=model_type
            )
            results[seq_len] = metrics
        except RuntimeError as e:
            print(f"Error evaluating seq_len={seq_len}: {e}")
            continue
    
    model.train()
    return results


@torch.no_grad()
def evaluate_ood_robustness(
    model,
    model_type: str = "simple_regression",
    n_batches: int = 10
) -> Dict[str, EvaluationMetrics]:
    """
    Evaluate out-of-distribution robustness (Figure 4 in paper).
    Tests: skewed covariance, noisy labels, different orthants.
    
    Args:
        model: The transformer model
        model_type: Type of model
        n_batches: Number of test batches
    
    Returns:
        Dictionary of OOD scenario names to metrics
    """
    from data_sampler import generate_linear
    from config import n_points
    
    if model_type != "simple_regression":
        raise NotImplementedError("OOD evaluation currently only for linear regression")
    
    results = {}
    model.eval()
    device = get_device(model)
    
    # 1. Standard (in-distribution)
    make_batch_fn = lambda: generate_linear(n_points, batch_size, n_dims)
    results['standard'] = evaluate_model(
        model, n_batches, make_batch_fn, compute_baselines=True, model_type=model_type
    )
    
    # 2. Skewed covariance
    def generate_skewed():
        xs, ys = generate_linear(n_points, batch_size, n_dims)
        # Apply skewing transformation
        scale = torch.exp(torch.linspace(0, 2, n_dims))
        xs = xs * scale.unsqueeze(0).unsqueeze(0)
        return xs, ys
    
    results['skewed_covariance'] = evaluate_model(
        model, n_batches, generate_skewed, compute_baselines=True, model_type=model_type
    )
    
    # 3. Noisy labels (high noise in context, clean query)
    def generate_noisy():
        xs, ys = generate_linear(n_points, batch_size, n_dims)
        # Add large noise to all but last y
        noise = torch.randn_like(ys[:, :-1, :]) * 0.5
        ys[:, :-1, :] = ys[:, :-1, :] + noise
        return xs, ys
    
    results['noisy_labels'] = evaluate_model(
        model, n_batches, generate_noisy, compute_baselines=True, model_type=model_type
    )
    
    # 4. Different orthants (context positive, query can be negative)
    def generate_orthant():
        xs, ys = generate_linear(n_points, batch_size, n_dims)
        # Make context examples positive
        xs[:, :-1, :] = torch.abs(xs[:, :-1, :])
        # Query can be any sign (already is from original generation)
        return xs, ys
    
    results['different_orthants'] = evaluate_model(
        model, n_batches, generate_orthant, compute_baselines=True, model_type=model_type
    )
    
    model.train()
    return results


@torch.no_grad()
def evaluate_per_position_learning(
    model,
    make_batch_fn: Callable,
    n_batches: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze how loss decreases with more in-context examples.
    Returns mean and std of loss at each position (Figure 2 style).
    
    Args:
        model: The transformer model
        make_batch_fn: Function to generate batches
        n_batches: Number of batches to evaluate
    
    Returns:
        Tuple of (mean_losses, std_losses) arrays of shape (N,)
    """
    model.eval()
    device = get_device(model)
    
    all_position_losses = []
    
    for _ in range(n_batches):
        xs, ys = make_batch_fn()
        xs, ys = xs.to(device), ys.to(device)
        
        preds = model(xs, ys)
        
        B, N, output_dim = ys.shape
        y_pos = torch.arange(1, 2 * N, 2, device=device)
        pred_at_y = preds[:, y_pos, :]
        
        # Compute loss at each position
        position_losses = []
        for i in range(N):
            pos_loss = F.mse_loss(
                pred_at_y[:, i, :],
                ys[:, i, :],
                reduction='none'
            ).mean(dim=(0, 1)).item()
            position_losses.append(pos_loss)
        
        all_position_losses.append(position_losses)
    
    model.train()
    
    all_position_losses = np.array(all_position_losses)  # (n_batches, N)
    mean_losses = np.mean(all_position_losses, axis=0)
    std_losses = np.std(all_position_losses, axis=0)
    
    return mean_losses, std_losses


def print_evaluation_report(
    metrics: EvaluationMetrics,
    title: str = "Evaluation Report",
    verbose: bool = True
):
    """
    Print formatted evaluation report.
    
    Args:
        metrics: EvaluationMetrics object
        title: Report title
        verbose: Whether to print detailed statistics
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    
    print(f"\n{'Overall Metrics':^70}")
    print(f"{'-'*70}")
    print(f"  Total Loss:      {metrics.mean_total_loss:.6f} ± {metrics.std_total_loss:.6f}")
    print(f"  Query Loss:      {metrics.mean_query_loss:.6f} ± {metrics.std_query_loss:.6f}")
    print(f"  In-Context Loss: {metrics.mean_incontext_loss:.6f} ± {metrics.std_incontext_loss:.6f}")
    
    if metrics.baseline_comparisons:
        print(f"\n{'Baseline Comparisons':^70}")
        print(f"{'-'*70}")
        for name, value in metrics.baseline_comparisons.items():
            print(f"  {name}: {value:.6f}")
            if 'query' in name.lower():
                ratio = metrics.mean_query_loss / (value + 1e-10)
                print(f"  → Model/Baseline Ratio: {ratio:.4f}")
    
    if verbose and metrics.per_position_losses is not None:
        print(f"\n{'Per-Position Losses':^70}")
        print(f"{'-'*70}")
        for i, loss in enumerate(metrics.per_position_losses):
            print(f"  Position {i:2d}: {loss:.6f}")
    
    print(f"{'='*70}\n")


def print_ood_report(ood_results: Dict[str, EvaluationMetrics]):
    """Print out-of-distribution robustness report."""
    print(f"\n{'='*70}")
    print(f"{'Out-of-Distribution Robustness Analysis':^70}")
    print(f"{'='*70}\n")
    
    standard_loss = ood_results['standard'].mean_query_loss
    
    for scenario, metrics in ood_results.items():
        print(f"\n{scenario.upper().replace('_', ' ')}:")
        print(f"  Query Loss: {metrics.mean_query_loss:.6f} ± {metrics.std_query_loss:.6f}")
        
        if metrics.baseline_comparisons:
            baseline = metrics.baseline_comparisons.get('least_squares_query_loss', 0)
            print(f"  Baseline:   {baseline:.6f}")
            print(f"  Model/Baseline: {metrics.mean_query_loss / (baseline + 1e-10):.4f}")
        
        if scenario != 'standard':
            degradation = (metrics.mean_query_loss / standard_loss - 1) * 100
            print(f"  Degradation from standard: {degradation:+.2f}%")
    
    print(f"\n{'='*70}\n")


def print_generalization_report(gen_results: Dict[int, EvaluationMetrics]):
    """Print sequence length generalization report."""
    print(f"\n{'='*70}")
    print(f"{'Sequence Length Generalization':^70}")
    print(f"{'='*70}\n")
    print(f"{'Length':<10} {'Query Loss':<20} {'Baseline':<20} {'Ratio':<10}")
    print(f"{'-'*70}")
    
    for seq_len in sorted(gen_results.keys()):
        metrics = gen_results[seq_len]
        baseline_str = ""
        ratio_str = ""
        
        if metrics.baseline_comparisons:
            baseline = metrics.baseline_comparisons.get('least_squares_query_loss', 0)
            baseline_str = f"{baseline:.6f}"
            ratio_str = f"{metrics.mean_query_loss / (baseline + 1e-10):.4f}"
        
        print(f"{seq_len:<10} {metrics.mean_query_loss:.6f} ± {metrics.std_query_loss:.4f}  "
              f"{baseline_str:<20} {ratio_str:<10}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    """Example usage and testing."""
    from model import TransformerModel
    from data_sampler import generate_linear
    from config import n_points
    
    print("Initializing model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("\n1. Standard Evaluation")
    make_batch_fn = lambda: generate_linear(n_points, batch_size, n_dims)
    metrics = evaluate_model(
        model,
        n_test_batches=10,
        make_batch_fn=make_batch_fn,
        compute_baselines=True,
        model_type="simple_regression"
    )
    print_evaluation_report(metrics, "Standard Evaluation")
    
    print("\n2. Sequence Length Generalization")
    gen_results = evaluate_sequence_length_generalization(
        model,
        sequence_lengths=[5, 10, 15, 20, 25, 30],
        model_type="simple_regression",
        n_batches=5
    )
    print_generalization_report(gen_results)
    
    print("\n3. Out-of-Distribution Robustness")
    ood_results = evaluate_ood_robustness(
        model,
        model_type="simple_regression",
        n_batches=5
    )
    print_ood_report(ood_results)
