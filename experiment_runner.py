"""
Experiment runner for in-context learning transformer experiments.
Based on "What Can Transformers Learn In-Context?" (Garg et al., 2022)
https://arxiv.org/abs/2208.01066
"""

import torch
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from model import TransformerModel, NNTransformer
from data_sampler import generate_linear, generate_nn
from eval import (
    evaluate_model,
    evaluate_sequence_length_generalization,
    evaluate_ood_robustness,
    evaluate_per_position_learning,
    print_evaluation_report,
    print_generalization_report,
    print_ood_report
)
from visualization import (
    plot_training_curves,
    plot_incontext_learning_curve,
    plot_generalization_curves,
    plot_baseline_comparison,
    plot_ood_robustness,
    plot_predictions_vs_targets,
    create_summary_dashboard
)
from config import (
    lr, batch_size, n_dims, n_points, train_steps, log_every, eval_every,
    nn_hidden_dim, nn_input_dims, nn_output_dim
)


class ExperimentRunner:
    """Main experiment orchestrator for ICL transformer research."""
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "experiments",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize experiment runner.
        
        Args:
            experiment_name: Name for this experiment
            base_dir: Base directory for all experiments
            device: Device to run on
        """
        self.experiment_name = experiment_name
        self.device = device
        
        # Create experiment directory structure
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{experiment_name}_{self.timestamp}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.plot_dir = os.path.join(self.exp_dir, "plots")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        
        for directory in [self.exp_dir, self.checkpoint_dir, self.plot_dir, self.log_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Training history
        self.training_history = {
            'losses': [],
            'eval_steps': [],
            'eval_metrics': []
        }
        
        print(f"Experiment initialized: {self.exp_dir}")
    
    def train_model(
        self,
        model,
        model_type: str,
        train_steps: int = 2000,
        log_every: int = 50,
        eval_every: Optional[int] = None,
        save_checkpoints: bool = True
    ) -> Dict:
        """
        Train a model with logging and periodic evaluation.
        
        Args:
            model: The model to train
            model_type: Type of model ("simple_regression" or "nn")
            train_steps: Number of training steps
            log_every: Log loss every N steps
            eval_every: Evaluate every N steps (None to disable)
            save_checkpoints: Whether to save checkpoints
        
        Returns:
            Training history dictionary
        """
        from train_NN import train_step, generate_data
        
        print(f"\n{'='*70}")
        print(f"Training {model_type} model")
        print(f"{'='*70}")
        print(f"Steps: {train_steps}, Log every: {log_every}, Eval every: {eval_every}")
        print(f"Device: {self.device}\n")
        
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Create data generator
        if model_type == "simple_regression":
            make_batch_fn = lambda: generate_linear(n_points, batch_size, n_dims)
        elif model_type == "nn":
            make_batch_fn = lambda: generate_nn(
                n_points, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Training loop
        for step in range(train_steps):
            xs, ys = make_batch_fn()
            xs, ys = xs.to(self.device), ys.to(self.device)
            
            loss = train_step(model, xs, ys, optimizer)
            self.training_history['losses'].append(loss)
            
            if step % log_every == 0:
                print(f"Step {step:5d} | Loss: {loss:.6f}")
            
            # Periodic evaluation
            if eval_every and step > 0 and step % eval_every == 0:
                print(f"\n--- Evaluation at step {step} ---")
                metrics = evaluate_model(
                    model,
                    n_test_batches=10,
                    make_batch_fn=make_batch_fn,
                    compute_baselines=(model_type == "simple_regression"),
                    model_type=model_type
                )
                
                self.training_history['eval_steps'].append(step)
                self.training_history['eval_metrics'].append(metrics)
                
                print(f"Query Loss: {metrics.mean_query_loss:.6f} ± {metrics.std_query_loss:.6f}")
                
                if metrics.baseline_comparisons:
                    baseline = metrics.baseline_comparisons['least_squares_query_loss']
                    print(f"Baseline: {baseline:.6f}, Ratio: {metrics.mean_query_loss/baseline:.4f}")
                print()
            
            # Save checkpoint
            if save_checkpoints and step > 0 and step % (train_steps // 5) == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_step_{step}.pt"
                )
                self.save_checkpoint(model, optimizer, step, checkpoint_path, model_type)
        
        print(f"\nTraining complete!")
        
        # Save final model
        final_path = os.path.join(self.checkpoint_dir, "final_model.pt")
        self.save_checkpoint(model, optimizer, train_steps, final_path, model_type)
        
        return self.training_history
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        step: int,
        path: str,
        model_type: str
    ):
        """Save model checkpoint."""
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_type': model_type,
            'training_history': self.training_history,
            'config': {
                'n_dims': n_dims if model_type == "simple_regression" else nn_input_dims,
                'n_points': n_points,
                'lr': lr,
                'batch_size': batch_size
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Tuple[torch.nn.Module, Dict]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        model_type = checkpoint['model_type']
        config = checkpoint['config']
        
        if model_type == "simple_regression":
            model = TransformerModel(
                config['n_dims'],
                config['n_points'],
                name="simple_regression"
            )
        elif model_type == "nn":
            model = NNTransformer(
                config['n_dims'],
                config['n_points'],
                name="nn"
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model, checkpoint
    
    def run_full_evaluation(
        self,
        model,
        model_type: str,
        include_ood: bool = True,
        include_generalization: bool = True
    ) -> Dict:
        """
        Run comprehensive evaluation suite.
        
        Args:
            model: Trained model
            model_type: Type of model
            include_ood: Whether to run OOD robustness tests
            include_generalization: Whether to run generalization tests
        
        Returns:
            Dictionary of all evaluation results
        """
        print(f"\n{'='*70}")
        print(f"Running Full Evaluation Suite")
        print(f"{'='*70}\n")
        
        results = {}
        
        # Create data generator
        if model_type == "simple_regression":
            make_batch_fn = lambda: generate_linear(n_points, batch_size, n_dims)
        elif model_type == "nn":
            make_batch_fn = lambda: generate_nn(
                n_points, batch_size, nn_hidden_dim, nn_output_dim, nn_input_dims
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 1. Standard evaluation
        print("1. Standard Evaluation...")
        standard_metrics = evaluate_model(
            model,
            n_test_batches=20,
            make_batch_fn=make_batch_fn,
            compute_baselines=(model_type == "simple_regression"),
            model_type=model_type
        )
        results['standard'] = standard_metrics
        print_evaluation_report(standard_metrics, "Standard Evaluation")
        
        # 2. Per-position learning curve
        print("2. Per-Position Learning Analysis...")
        mean_losses, std_losses = evaluate_per_position_learning(
            model, make_batch_fn, n_batches=20
        )
        results['per_position'] = {
            'mean': mean_losses,
            'std': std_losses
        }
        
        # 3. Sequence length generalization
        if include_generalization:
            print("3. Sequence Length Generalization...")
            # Around line 282, change the generalization test to use safer defaults
            gen_results = evaluate_sequence_length_generalization(
                model,
                sequence_lengths=[5, 10, 15, 20],  # Changed from [5, 10, 15, 20, 25, 30, 40, 50]
                model_type=model_type,
                n_batches=10
            )

            results['generalization'] = gen_results
            print_generalization_report(gen_results)
        
        # 4. OOD robustness
        if include_ood and model_type == "simple_regression":
            print("4. Out-of-Distribution Robustness...")
            ood_results = evaluate_ood_robustness(
                model,
                model_type=model_type,
                n_batches=10
            )
            results['ood'] = ood_results
            print_ood_report(ood_results)
        
        return results
    
    def generate_visualizations(
        self,
        model,
        model_type: str,
        eval_results: Dict
    ):
        """
        Generate all visualizations for the experiment.
        
        Args:
            model: Trained model
            model_type: Type of model
            eval_results: Results from run_full_evaluation
        """
        print(f"\n{'='*70}")
        print(f"Generating Visualizations")
        print(f"{'='*70}\n")
        
        # 1. Training curves
        if self.training_history['losses']:
            print("1. Training curves...")
            plot_training_curves(
                self.training_history['losses'],
                eval_steps=self.training_history.get('eval_steps'),
                eval_metrics=self.training_history.get('eval_metrics'),
                save_path=os.path.join(self.plot_dir, "training_curves.png")
            )
        
        # 2. In-context learning curve (per-position)
        if 'per_position' in eval_results:
            print("2. In-context learning curve...")
            mean_losses = eval_results['per_position']['mean']
            std_losses = eval_results['per_position']['std']
            
            # Compute baseline if linear
            baseline = None
            if model_type == "simple_regression":
                from data_sampler import generate_linear
                xs, ys = generate_linear(n_points, batch_size, n_dims)
                baseline_loss = []
                for i in range(1, n_points + 1):
                    if i == 1:
                        baseline_loss.append(mean_losses[0])  # No context
                    else:
                        # Approximate LS baseline at each position
                        baseline_loss.append(1.0 / i)  # Theoretical scaling
                baseline = np.array(baseline_loss)
            
            plot_incontext_learning_curve(
                mean_losses,
                std_losses,
                baseline=baseline,
                save_path=os.path.join(self.plot_dir, "incontext_learning.png")
            )
        
        # 3. Generalization curves
        if 'generalization' in eval_results:
            print("3. Generalization curves...")
            plot_generalization_curves(
                eval_results['generalization'],
                save_path=os.path.join(self.plot_dir, "generalization.png")
            )
        
        # 4. Baseline comparison
        if model_type == "simple_regression" and 'standard' in eval_results:
            print("4. Baseline comparison...")
            if eval_results['standard'].baseline_comparisons:
                plot_baseline_comparison(
                    eval_results['standard'],
                    save_path=os.path.join(self.plot_dir, "baseline_comparison.png")
                )
        
        # 5. OOD robustness
        if 'ood' in eval_results:
            print("5. OOD robustness...")
            plot_ood_robustness(
                eval_results['ood'],
                save_path=os.path.join(self.plot_dir, "ood_robustness.png")
            )
        
        # 6. Predictions vs targets
        print("6. Predictions vs targets...")
        plot_predictions_vs_targets(
            model,
            n_examples=3,
            model_type=model_type,
            save_path=os.path.join(self.plot_dir, "predictions.png")
        )
        
        # 7. Summary dashboard
        print("7. Summary dashboard...")
        create_summary_dashboard(
            training_history=self.training_history,
            eval_results=eval_results,
            model_type=model_type,
            save_path=os.path.join(self.plot_dir, "summary_dashboard.png")
        )
        
        print(f"\nAll visualizations saved to: {self.plot_dir}")
    
    def save_results(self, eval_results: Dict, filename: str = "results.json"):
        """Save evaluation results to JSON."""
        # Convert numpy arrays and custom objects to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(eval_results)
        
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {path}")
    
    def run_experiment(
        self,
        model_type: str = "simple_regression",
        train_steps: int = 2000,
        run_evaluation: bool = True,
        generate_plots: bool = True
    ):
        """
        Run a complete experiment: train, evaluate, visualize.
        
        Args:
            model_type: Type of model to train
            train_steps: Number of training steps
            run_evaluation: Whether to run full evaluation
            generate_plots: Whether to generate visualizations
        """
        print(f"\n{'#'*70}")
        print(f"# Experiment: {self.experiment_name}")
        print(f"# Model Type: {model_type}")
        print(f"# Training Steps: {train_steps}")
        print(f"{'#'*70}\n")
        
        # Create model
        if model_type == "simple_regression":
            model = TransformerModel(n_dims, n_points, name="simple_regression")
        elif model_type == "nn":
            model = NNTransformer(nn_input_dims, n_points, name="nn")
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Train
        training_history = self.train_model(
            model,
            model_type,
            train_steps=train_steps,
            log_every=log_every,
            eval_every=eval_every
        )
        
        # Evaluate
        eval_results = None
        if run_evaluation:
            eval_results = self.run_full_evaluation(
                model,
                model_type,
                include_ood=(model_type == "simple_regression"),
                include_generalization=True
            )
            self.save_results(eval_results)
        
        # Visualize
        if generate_plots and eval_results:
            self.generate_visualizations(model, model_type, eval_results)
        
        # Print summary
        print(f"\n{'#'*70}")
        print(f"# Experiment Complete!")
        print(f"# Results saved to: {self.exp_dir}")
        print(f"{'#'*70}\n")
        
        return model, eval_results


def compare_models(
    experiment_name: str,
    model_types: List[str] = ["simple_regression", "nn"],
    train_steps: int = 2000
):
    """
    Compare multiple model types in a single experiment.
    
    Args:
        experiment_name: Name for this comparison experiment
        model_types: List of model types to compare
        train_steps: Number of training steps for each model
    """
    runner = ExperimentRunner(f"{experiment_name}_comparison")
    
    all_results = {}
    all_models = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type} model...")
        print(f"{'='*70}\n")
        
        # Create model
        if model_type == "simple_regression":
            model = TransformerModel(n_dims, n_points, name="simple_regression")
        elif model_type == "nn":
            model = NNTransformer(nn_input_dims, n_points, name="nn")
        else:
            continue
        
        # Train
        runner.train_model(model, model_type, train_steps=train_steps)
        
        # Evaluate
        eval_results = runner.run_full_evaluation(
            model,
            model_type,
            include_ood=(model_type == "simple_regression")
        )
        
        all_results[model_type] = eval_results
        all_models[model_type] = model
    
    # Generate comparison visualizations
    runner.save_results(all_results, "comparison_results.json")
    
    print(f"\nModel comparison complete! Results in: {runner.exp_dir}")
    
    return all_models, all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ICL transformer experiments")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="icl_experiment",
        help="Name for this experiment"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="simple_regression",
        choices=["simple_regression", "nn", "compare"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=2000,
        help="Number of training steps"
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Skip evaluation"
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Skip plot generation"
    )
    
    args = parser.parse_args()
    
    if args.model_type == "compare":
        compare_models(
            args.experiment_name,
            model_types=["simple_regression", "nn"],
            train_steps=args.train_steps
        )
    else:
        runner = ExperimentRunner(args.experiment_name)
        runner.run_experiment(
            model_type=args.model_type,
            train_steps=args.train_steps,
            run_evaluation=not args.no_eval,
            generate_plots=not args.no_plots
        )
