"""
Comprehensive generalization testing script.
Tests model performance on various out-of-distribution scenarios.
"""

import torch
from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from config import (
    batch_size, n_dims, n_points, 
    nn_hidden_dim, nn_input_dims, nn_output_dim
)
from eval import evaluate_generalization, evaluate_few_shot_performance, evaluate_out_of_distribution
from visualization import plot_generalization_curves, plot_few_shot_curves
import json


def test_sequence_length_generalization(model, model_type="simple_regression", 
                                       sequence_lengths=None, n_batches=10):
    """
    Test how well the model generalizes to different sequence lengths.
    """
    if sequence_lengths is None:
        sequence_lengths = [5, 10, 15, 20, 25, 30, 40, 50]
    
    print(f"\n{'='*60}")
    print("Testing Sequence Length Generalization")
    print(f"{'='*60}")
    
    results = evaluate_generalization(
        model, 
        sequence_lengths=sequence_lengths,
        model_type=model_type,
        n_batches=n_batches
    )
    
    print("\nResults:")
    for seq_len in sorted(results.keys()):
        result = results[seq_len]
        print(f"  Length {seq_len:3d}: Loss = {result['mean_query_loss']:.6f} ± {result['std_query_loss']:.6f}")
    
    # Plot results
    plot_generalization_curves(results, save_path=f"generalization_seq_length_{model_type}.png")
    
    return results


def test_few_shot_generalization(model, model_type="simple_regression", 
                                 k_shots=None, n_batches=10):
    """
    Test how well the model performs with different numbers of in-context examples.
    """
    if k_shots is None:
        k_shots = [1, 2, 3, 5, 10, 15, 20]
    
    print(f"\n{'='*60}")
    print("Testing Few-Shot Generalization")
    print(f"{'='*60}")
    
    results = evaluate_few_shot_performance(
        model,
        k_shots=k_shots,
        model_type=model_type,
        n_batches=n_batches
    )
    
    print("\nResults:")
    for k in sorted(results.keys()):
        result = results[k]
        print(f"  {k:2d}-shot: Loss = {result['mean_query_loss']:.6f} ± {result['std_query_loss']:.6f}")
    
    # Plot results
    plot_few_shot_curves(results, save_path=f"generalization_few_shot_{model_type}.png")
    
    return results


def test_noise_robustness(model, model_type="simple_regression", 
                         noise_levels=None, n_batches=10):
    """
    Test model robustness to different noise levels.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    print(f"\n{'='*60}")
    print("Testing Noise Robustness")
    print(f"{'='*60}")
    
    results = evaluate_out_of_distribution(
        model,
        noise_levels=noise_levels,
        model_type=model_type,
        n_batches=n_batches
    )
    
    print("\nResults:")
    for noise in sorted(results.keys()):
        result = results[noise]
        print(f"  Noise {noise:.2f}: Loss = {result['mean_query_loss']:.6f} ± {result['std_query_loss']:.6f}")
    
    return results


def test_dimensionality_generalization(model, model_type="simple_regression", 
                                       input_dims_list=None, n_batches=5):
    """
    Test how well the model generalizes to different input dimensions.
    Note: This requires retraining or using a model that can handle variable dimensions.
    """
    if model_type != "simple_regression":
        print("Dimensionality generalization test only supported for simple_regression")
        return None
    
    if input_dims_list is None:
        input_dims_list = [4, 6, 8, 10, 12, 16]
    
    print(f"\n{'='*60}")
    print("Testing Dimensionality Generalization")
    print(f"{'='*60}")
    print("Note: This test assumes the model can handle variable dimensions")
    
    results = {}
    
    model.eval()
    with torch.no_grad():
        for dim in input_dims_list:
            query_losses = []
            
            for _ in range(n_batches):
                xs = torch.randn(batch_size, n_points, dim)
                w = torch.randn(batch_size, dim, 1)
                ys = xs @ w + 0.1 * torch.randn(batch_size, n_points, 1)
                
                # Note: This will fail if model was trained on fixed n_dims
                # You may need to pad/truncate or use a different approach
                try:
                    preds = model(xs, ys)
                    B, N, _ = ys.shape
                    y_pos = torch.arange(1, 2*N, 2, device=preds.device)
                    pred_all = preds[:, y_pos, :].squeeze(-1)
                    tgt_all = ys.squeeze(-1)
                    
                    query_pred = pred_all[:, -1]
                    query_tgt = tgt_all[:, -1]
                    query_loss = torch.nn.functional.mse_loss(query_pred, query_tgt)
                    query_losses.append(query_loss.item())
                except Exception as e:
                    print(f"  Dimension {dim}: Failed - {e}")
                    query_losses = None
                    break
            
            if query_losses:
                results[dim] = {
                    "mean_query_loss": sum(query_losses) / len(query_losses),
                    "std_query_loss": (sum([(x - sum(query_losses)/len(query_losses))**2 
                                           for x in query_losses]) / len(query_losses))**0.5,
                }
                print(f"  Dim {dim:2d}: Loss = {results[dim]['mean_query_loss']:.6f} ± {results[dim]['std_query_loss']:.6f}")
    
    model.train()
    return results


def run_comprehensive_generalization_tests(model_path=None, model_type="simple_regression"):
    """
    Run all generalization tests and save results.
    
    Args:
        model_path: Path to saved model (optional, will create new model if None)
        model_type: "simple_regression" or "nn"
    """
    # Load or create model
    if model_path:
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path)
        if model_type == "simple_regression":
            model = TransformerModel(n_dims, n_points, name="simple_regression")
        else:
            model = NNTransformer(nn_input_dims, n_points, name="nn")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Creating new model (untrained)...")
        if model_type == "simple_regression":
            model = TransformerModel(n_dims, n_points, name="simple_regression")
        else:
            model = NNTransformer(nn_input_dims, n_points, name="nn")
    
    all_results = {}
    
    # Test sequence length generalization
    seq_results = test_sequence_length_generalization(model, model_type)
    all_results['sequence_length'] = seq_results
    
    # Test few-shot generalization
    few_shot_results = test_few_shot_generalization(model, model_type)
    all_results['few_shot'] = few_shot_results
    
    # Test noise robustness
    noise_results = test_noise_robustness(model, model_type)
    all_results['noise'] = noise_results
    
    # Save all results
    results_file = f"generalization_results_{model_type}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run generalization tests")
    parser.add_argument("--model_type", type=str, default="simple_regression",
                       choices=["simple_regression", "nn"],
                       help="Type of model to test")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to saved model checkpoint")
    
    args = parser.parse_args()
    
    run_comprehensive_generalization_tests(
        model_path=args.model_path,
        model_type=args.model_type
    )

