import torch
import os
from model import TransformerModel, NNTransformer
from train_NN import train
from data_sampler import generate_linear, generate_nn
from eval import evaluate_model, evaluate_generalization, print_evaluation_report
from visualization import plot_predictions_vs_targets, plot_generalization_curves
from config import (
    lr, batch_size, n_dims, n_points, train_steps, log_every,
    nn_hidden_dim, nn_input_dims, nn_output_dim
)


def train_and_evaluate(model_type="simple_regression", save_model=True):
    """Train a model and run evaluation."""
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
    
    print(model_type)
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
    
    if model_type == "simple_regression":

        make_batch_fn = lambda: generate_linear(n_points, batch_size, n_dims)

    elif model_type == "nn":

        make_batch_fn = lambda: generate_nn(n_points, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims)

 

    metrics = evaluate_model(

        model,

        n_test_batches=10,

        make_batch_fn=make_batch_fn)
    
    print(metrics)
    
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
    
    # Plot results
    os.makedirs("plots", exist_ok=True)
    plot_generalization_curves(gen_results, 
                              save_path=f"plots/gen_seq_length_{model_type}.png")
    
    # Visualize predictions
    print(f"\n{'='*70}")
    print("Generating Visualizations")
    print(f"{'='*70}\n")
    
    plot_predictions_vs_targets(
        model,
        n_examples=3,
        model_type=model_type,
        save_path=f"plots/predictions_{model_type}.png"
    )
    
    print(f"\n{'='*70}")
    print("Experiment Complete!")
    print(f"{'='*70}\n")
    print("Results saved in checkpoints/ and plots/")
    
    return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ICL transformer experiments")
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple_regression",
        choices=["simple_regression", "nn"],
        help="Type of model"
    )
    args = parser.parse_args()
    
    train_and_evaluate(model_type=args.model_type)
