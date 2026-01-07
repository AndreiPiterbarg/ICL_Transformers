"""
Comprehensive experiment runner for ICL transformer experiments.
This script ties together training, evaluation, and analysis.
"""

import torch
import argparse
import os
from pathlib import Path
from model import TransformerModel, NNTransformer
from train_NN import train, generate_data
from eval import (
    evaluate_model, evaluate_generalization, evaluate_few_shot_performance,
    print_evaluation_report, save_evaluation_results
)
from attention_analysis import (
    extract_attention_weights, visualize_attention_patterns,
    analyze_attention_to_query, analyze_attention_entropy
)
from visualization import (
    plot_predictions_vs_targets, plot_error_distribution,
    plot_generalization_curves, plot_few_shot_curves
)
from generalization_test import run_comprehensive_generalization_tests
from config import (
    lr, batch_size, n_dims, n_points, train_steps, log_every,
    nn_hidden_dim, nn_input_dims, nn_output_dim
)


def train_and_evaluate(model_type="simple_regression", save_model=True, 
                      eval_during_training=True, run_analysis=True):
    """
    Train a model and run comprehensive evaluation and analysis.
    
    Args:
        model_type: "simple_regression" or "nn"
        save_model: Whether to save the trained model
        eval_during_training: Whether to evaluate during training
        run_analysis: Whether to run attention analysis after training
    """
    print(f"\n{'='*70}")
    print(f"Training {model_type} Model")
    print(f"{'='*70}\n")
    
    # Create model
    if model_type == "simple_regression":
        model = TransformerModel(n_dims, n_points, name="simple_regression")
    elif model_type == "nn":
        model = NNTransformer(nn_input_dims, n_points, name="nn")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Train model
    print("Starting training...")
    train(model, train_steps=train_steps, log_every=log_every)
    
    # Save model
    if save_model:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/model_{model_type}_final.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'n_dims': n_dims if model_type == "simple_regression" else nn_input_dims,
            'n_points': n_points,
        }, checkpoint_path)
        print(f"\nModel saved to {checkpoint_path}")
    
    # Evaluate model
    print(f"\n{'='*70}")
    print("Evaluating Model")
    print(f"{'='*70}\n")
    
    metrics = evaluate_model(model, n_test_batches=20, model_type=model_type)
    print_evaluation_report(metrics, f"{model_type.upper()} Evaluation")
    
    # Save evaluation results
    os.makedirs("results", exist_ok=True)
    save_evaluation_results(metrics, f"results/eval_{model_type}.json")
    
    # Run generalization tests
    print(f"\n{'='*70}")
    print("Running Generalization Tests")
    print(f"{'='*70}\n")
    
    gen_results = evaluate_generalization(
        model,
        sequence_lengths=[5, 10, 15, 20, 25, 30],
        model_type=model_type,
        n_batches=10
    )
    
    few_shot_results = evaluate_few_shot_performance(
        model,
        k_shots=[1, 3, 5, 10, 15],
        model_type=model_type,
        n_batches=10
    )
    
    # Plot results
    os.makedirs("plots", exist_ok=True)
    plot_generalization_curves(gen_results, 
                              save_path=f"plots/gen_seq_length_{model_type}.png")
    plot_few_shot_curves(few_shot_results,
                        save_path=f"plots/gen_few_shot_{model_type}.png")
    
    # Run attention analysis
    if run_analysis:
        print(f"\n{'='*70}")
        print("Running Attention Analysis")
        print(f"{'='*70}\n")
        
        from data_sampler import generate_linear, generate_nn
        
        if model_type == "simple_regression":
            xs, ys = generate_linear(n_points, batch_size=1, n_dims=n_dims)
        else:
            xs, ys = generate_nn(n_points, batch_size=1, nn_hidden_dim, 
                                nn_output_dim, nn_input_dims)
        
        attention_weights = extract_attention_weights(model, xs, ys)
        
        # Visualize attention patterns
        visualize_attention_patterns(
            attention_weights,
            layer_idx=-1,
            head_idx=0,
            batch_idx=0,
            save_path=f"plots/attention_last_layer_{model_type}.png",
            title=f"Last Layer Attention - {model_type}"
        )
        
        # Analyze attention to query
        query_analysis = analyze_attention_to_query(attention_weights, layer_idx=-1)
        print("Attention to Query Analysis:")
        print(f"  X attention: {query_analysis['x_attention']:.4f}")
        print(f"  Y attention: {query_analysis['y_attention']:.4f}")
        print(f"  Recent attention: {query_analysis['recent_attention']:.4f}")
        
        # Calculate entropy
        entropies = analyze_attention_entropy(attention_weights)
        print("\nAttention Entropy:")
        for layer_key, stats in entropies.items():
            print(f"  {layer_key}: Mean = {stats['mean_entropy']:.4f}, "
                  f"Query = {stats['query_entropy']:.4f}")
    
    # Visualize predictions
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}\n")
    
    plot_predictions_vs_targets(
        model,
        n_examples=5,
        model_type=model_type,
        save_path=f"plots/predictions_{model_type}.png"
    )
    
    plot_error_distribution(
        metrics,
        save_path=f"plots/error_distribution_{model_type}.png"
    )
    
    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
    print("Results saved in:")
    print("  - checkpoints/")
    print("  - results/")
    print("  - plots/")
    
    return model, metrics


def load_and_evaluate(model_path, model_type="simple_regression"):
    """
    Load a saved model and run evaluation.
    
    Args:
        model_path: Path to saved model checkpoint
        model_type: "simple_regression" or "nn"
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == "simple_regression":
        n_dims = checkpoint.get('n_dims', 8)
        n_points = checkpoint.get('n_points', 20)
        model = TransformerModel(n_dims, n_points, name="simple_regression")
    else:
        n_dims = checkpoint.get('n_dims', nn_input_dims)
        n_points = checkpoint.get('n_points', 20)
        model = NNTransformer(n_dims, n_points, name="nn")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Running evaluation...")
    metrics = evaluate_model(model, n_test_batches=20, model_type=model_type)
    print_evaluation_report(metrics, f"{model_type.upper()} Evaluation")
    
    return model, metrics


def compare_models(model_paths, model_types):
    """
    Compare multiple trained models.
    
    Args:
        model_paths: List of paths to model checkpoints
        model_types: List of model types corresponding to paths
    """
    print(f"\n{'='*70}")
    print("Comparing Models")
    print(f"{'='*70}\n")
    
    results = {}
    
    for path, mtype in zip(model_paths, model_types):
        print(f"\nEvaluating {path} ({mtype})...")
        model, metrics = load_and_evaluate(path, mtype)
        results[path] = {
            'type': mtype,
            'metrics': metrics
        }
    
    # Print comparison
    print(f"\n{'='*70}")
    print("Comparison Summary")
    print(f"{'='*70}\n")
    
    for path, result in results.items():
        metrics = result['metrics']
        print(f"{Path(path).stem} ({result['type']}):")
        print(f"  Query Loss: {metrics['mean_query_loss']:.6f} ± {metrics['std_query_loss']:.6f}")
        if 'query_rmse' in metrics:
            print(f"  Query RMSE: {metrics['query_rmse']:.6f}")
        print()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive ICL transformer experiments"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "compare"],
        help="Mode: train (train and evaluate), eval (evaluate saved model), compare (compare models)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple_regression",
        choices=["simple_regression", "nn"],
        help="Type of model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model (for eval mode)"
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to saved models (for compare mode)"
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=None,
        help="Model types corresponding to model_paths"
    )
    parser.add_argument(
        "--no_analysis",
        action="store_true",
        help="Skip attention analysis"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_and_evaluate(
            model_type=args.model_type,
            run_analysis=not args.no_analysis
        )
    elif args.mode == "eval":
        if args.model_path is None:
            print("Error: --model_path required for eval mode")
        else:
            load_and_evaluate(args.model_path, args.model_type)
    elif args.mode == "compare":
        if args.model_paths is None or args.model_types is None:
            print("Error: --model_paths and --model_types required for compare mode")
        else:
            compare_models(args.model_paths, args.model_types)

