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
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt


def evaluate_model(model, n_test_batches=10, model_type="simple_regression"):
    """
    Evaluate model on test data and return metrics.
    
    Args:
        model: The transformer model to evaluate
        n_test_batches: Number of test batches to evaluate on
        model_type: "simple_regression" or "nn"
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    query_losses = []  # Loss on the final query point
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for _ in range(n_test_batches):
            if model_type == "simple_regression":
                xs, ys = generate_linear(n_points, batch_size, n_dims)
            elif model_type == "nn":
                xs, ys = generate_nn(n_points, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            preds = model(xs, ys)  # (B, 2N, output_dim)
            B, N, output_dim = ys.shape
            
            # Get y positions in interleaved sequence
            y_pos = torch.arange(1, 2*N, 2, device=preds.device)
            
            # Extract predictions and targets
            pred_all = preds[:, y_pos, :]  # (B, N, output_dim)
            tgt_all = ys  # (B, N, output_dim)
            
            # Squeeze if output_dim == 1 for compatibility
            if output_dim == 1:
                pred_all = pred_all.squeeze(-1)  # (B, N)
                tgt_all = tgt_all.squeeze(-1)  # (B, N)
            
            # Overall loss
            if output_dim == 1:
                loss = F.mse_loss(pred_all, tgt_all)
            else:
                loss = F.mse_loss(pred_all, tgt_all)
            
            total_loss += loss.item()
            
            # Query loss (final point)
            query_pred = pred_all[:, -1]  # (B,) or (B, output_dim)
            query_tgt = tgt_all[:, -1]    # (B,) or (B, output_dim)
            query_loss = F.mse_loss(query_pred, query_tgt)
            query_losses.append(query_loss.item())
            
            all_predictions.append(pred_all.cpu())
            all_targets.append(tgt_all.cpu())
    
    model.train()
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = {
        "mean_loss": total_loss / n_test_batches,
        "mean_query_loss": np.mean(query_losses),
        "std_query_loss": np.std(query_losses),
        "predictions": all_predictions,
        "targets": all_targets,
    }
    
    # Additional metrics for regression
    if output_dim == 1:
        mae = torch.abs(all_predictions - all_targets).mean().item()
        rmse = torch.sqrt(F.mse_loss(all_predictions, all_targets)).item()
        metrics["mae"] = mae
        metrics["rmse"] = rmse
        
        # Query-specific metrics
        query_preds = all_predictions[:, -1]
        query_tgts = all_targets[:, -1]
        metrics["query_mae"] = torch.abs(query_preds - query_tgts).mean().item()
        metrics["query_rmse"] = torch.sqrt(F.mse_loss(query_preds, query_tgts)).item()
    
    return metrics


def evaluate_generalization(model, sequence_lengths: list, model_type="simple_regression", n_batches=5):
    """
    Evaluate model generalization to different sequence lengths.
    
    Args:
        model: The transformer model
        sequence_lengths: List of sequence lengths to test
        model_type: "simple_regression" or "nn"
        n_batches: Number of batches per sequence length
    
    Returns:
        Dictionary mapping sequence length to metrics
    """
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
                y_pos = torch.arange(1, 2*N, 2, device=preds.device)
                
                pred_all = preds[:, y_pos, :]  # (B, N, output_dim)
                tgt_all = ys  # (B, N, output_dim)
                
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


def evaluate_few_shot_performance(model, k_shots: list, model_type="simple_regression", n_batches=5):
    """
    Evaluate model performance with different numbers of in-context examples (few-shot).
    
    Args:
        model: The transformer model
        k_shots: List of k values (number of examples before query)
        model_type: "simple_regression" or "nn"
        n_batches: Number of batches per k
    
    Returns:
        Dictionary mapping k to metrics
    """
    results = {}
    
    for k in k_shots:
        if k >= n_points:
            continue  # Skip if k >= total points
        
        query_losses = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(n_batches):
                if model_type == "simple_regression":
                    xs_full, ys_full = generate_linear(n_points, batch_size, n_dims)
                elif model_type == "nn":
                    xs_full, ys_full = generate_nn(n_points, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                # Use only first k examples + query
                xs = xs_full[:, :k+1, :]
                ys = ys_full[:, :k+1, :]
                
                preds = model(xs, ys)
                B, N, output_dim = ys.shape
                y_pos = torch.arange(1, 2*N, 2, device=preds.device)
                
                pred_all = preds[:, y_pos, :]  # (B, N, output_dim)
                tgt_all = ys  # (B, N, output_dim)
                
                if output_dim == 1:
                    pred_all = pred_all.squeeze(-1)
                    tgt_all = tgt_all.squeeze(-1)
                
                query_pred = pred_all[:, -1]
                query_tgt = tgt_all[:, -1]
                query_loss = F.mse_loss(query_pred, query_tgt)
                query_losses.append(query_loss.item())
        
        model.train()
        results[k] = {
            "mean_query_loss": np.mean(query_losses),
            "std_query_loss": np.std(query_losses),
        }
    
    return results


def evaluate_out_of_distribution(model, noise_levels: list, model_type="simple_regression", n_batches=5):
    """
    Evaluate model robustness to different noise levels (OOD evaluation).
    
    Args:
        model: The transformer model
        noise_levels: List of noise levels to test
        model_type: "simple_regression" or "nn"
        n_batches: Number of batches per noise level
    
    Returns:
        Dictionary mapping noise level to metrics
    """
    results = {}
    
    for noise_level in noise_levels:
        query_losses = []
        
        model.eval()
        with torch.no_grad():
            for _ in range(n_batches):
                if model_type == "simple_regression":
                    xs = torch.randn(batch_size, n_points, n_dims)
                    w = torch.randn(batch_size, n_dims, 1)
                    ys = xs @ w + noise_level * torch.randn(batch_size, n_points, 1)
                elif model_type == "nn":
                    xs = torch.randn(batch_size, n_points, nn_input_dims)
                    ys = []
                    for i in range(batch_size):
                        x = xs[i]
                        hidden_layer = torch.nn.Linear(nn_input_dims, nn_hidden_dim)
                        output_layer = torch.nn.Linear(nn_hidden_dim, nn_output_dim)
                        pass1 = hidden_layer(x)
                        activated_pass1 = F.relu(pass1)
                        pass2 = output_layer(activated_pass1)
                        ys.append(pass2)
                    ys_out = torch.stack(ys)
                    ys = ys_out + noise_level * torch.randn_like(ys_out)
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")
                
                preds = model(xs, ys)
                B, N, output_dim = ys.shape
                y_pos = torch.arange(1, 2*N, 2, device=preds.device)
                
                pred_all = preds[:, y_pos, :]  # (B, N, output_dim)
                tgt_all = ys  # (B, N, output_dim)
                
                if output_dim == 1:
                    pred_all = pred_all.squeeze(-1)
                    tgt_all = tgt_all.squeeze(-1)
                
                query_pred = pred_all[:, -1]
                query_tgt = tgt_all[:, -1]
                query_loss = F.mse_loss(query_pred, query_tgt)
                query_losses.append(query_loss.item())
        
        model.train()
        results[noise_level] = {
            "mean_query_loss": np.mean(query_losses),
            "std_query_loss": np.std(query_losses),
        }
    
    return results


def print_evaluation_report(metrics: Dict, title: str = "Evaluation Report"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Mean Loss: {metrics['mean_loss']:.6f}")
    print(f"Mean Query Loss: {metrics['mean_query_loss']:.6f} ± {metrics['std_query_loss']:.6f}")
    
    if "mae" in metrics:
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"Query MAE: {metrics['query_mae']:.6f}")
        print(f"Query RMSE: {metrics['query_rmse']:.6f}")
    print(f"{'='*50}\n")


def save_evaluation_results(results: Dict, filepath: str):
    """Save evaluation results to a file."""
    import json
    # Convert tensors to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Loading model...")
    model = TransformerModel(n_dims, n_points, name="simple_regression")
    
    print("Evaluating model...")
    metrics = evaluate_model(model, n_test_batches=10, model_type="simple_regression")
    print_evaluation_report(metrics, "Standard Evaluation")
    
    print("Testing generalization to different sequence lengths...")
    gen_results = evaluate_generalization(
        model, 
        sequence_lengths=[5, 10, 15, 20, 25, 30],
        model_type="simple_regression",
        n_batches=5
    )
    print("\nGeneralization Results:")
    for seq_len, result in gen_results.items():
        print(f"  Sequence Length {seq_len}: Query Loss = {result['mean_query_loss']:.6f} ± {result['std_query_loss']:.6f}")
    
    print("\nTesting few-shot performance...")
    few_shot_results = evaluate_few_shot_performance(
        model,
        k_shots=[1, 3, 5, 10, 15],
        model_type="simple_regression",
        n_batches=5
    )
    print("\nFew-Shot Results:")
    for k, result in few_shot_results.items():
        print(f"  {k}-shot: Query Loss = {result['mean_query_loss']:.6f} ± {result['std_query_loss']:.6f}")

