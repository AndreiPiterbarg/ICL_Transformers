"""
Simple but effective curriculum learning for ICL Transformers.

This module implements a progressive curriculum that gradually increases task difficulty
by varying:
1. Number of in-context examples (sequence length)
2. Input dimensionality
3. Noise level in the data

The curriculum helps the model learn by starting with easier tasks and progressively
increasing difficulty, which can lead to:
- Faster convergence
- Better generalization
- More stable training
"""

import torch
import numpy as np
from typing import Dict, Tuple, Callable, Optional
from data_sampler import generate_linear, generate_nn


class CurriculumScheduler:
    """
    Manages curriculum progression during training.
    
    Uses a simple stage-based approach where difficulty increases at predefined steps.
    """
    
    def __init__(
        self,
        total_steps: int,
        num_stages: int = 4,
        start_points: int = 5,
        end_points: int = 20,
        start_dims: Optional[int] = None,
        end_dims: Optional[int] = None,
        start_noise: float = 0.05,
        end_noise: float = 0.1,
        warmup_ratio: float = 0.1
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            total_steps: Total number of training steps
            num_stages: Number of curriculum stages (default: 4)
            start_points: Initial number of in-context examples
            end_points: Final number of in-context examples
            start_dims: Initial dimensionality (None = use config default)
            end_dims: Final dimensionality (None = use config default)
            start_noise: Initial noise level
            end_noise: Final noise level
            warmup_ratio: Fraction of training to use for warmup (default: 0.1)
        """
        self.total_steps = total_steps
        self.num_stages = num_stages
        self.warmup_steps = int(total_steps * warmup_ratio)
        
        # Define curriculum parameters
        self.start_points = start_points
        self.end_points = end_points
        self.start_dims = start_dims
        self.end_dims = end_dims
        self.start_noise = start_noise
        self.end_noise = end_noise
        
        # Calculate stage boundaries
        remaining_steps = total_steps - self.warmup_steps
        self.stage_steps = remaining_steps // num_stages
        
        print(f"Curriculum Scheduler initialized:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {self.warmup_steps}")
        print(f"  Number of stages: {num_stages}")
        print(f"  Points progression: {start_points} → {end_points}")
        if start_dims and end_dims:
            print(f"  Dims progression: {start_dims} → {end_dims}")
        print(f"  Noise progression: {start_noise} → {end_noise}")
    
    def get_stage(self, step: int) -> int:
        """Get current curriculum stage (0 to num_stages-1)."""
        if step < self.warmup_steps:
            return 0
        stage = min(
            (step - self.warmup_steps) // self.stage_steps,
            self.num_stages - 1
        )
        return stage
    
    def get_curriculum_params(self, step: int) -> Dict:
        """
        Get curriculum parameters for the current step.
        
        Returns:
            Dictionary with 'n_points', 'n_dims', and 'noise_scale'
        """
        stage = self.get_stage(step)
        progress = stage / max(1, self.num_stages - 1)  # 0.0 to 1.0
        
        # Linear interpolation for each parameter
        n_points = int(
            self.start_points + (self.end_points - self.start_points) * progress
        )
        
        noise_scale = (
            self.start_noise + (self.end_noise - self.start_noise) * progress
        )
        
        params = {
            'n_points': n_points,
            'noise_scale': noise_scale,
            'stage': stage,
            'progress': progress
        }
        
        # Optional dimensionality progression
        if self.start_dims is not None and self.end_dims is not None:
            n_dims = int(
                self.start_dims + (self.end_dims - self.start_dims) * progress
            )
            params['n_dims'] = n_dims
        
        return params


class CurriculumDataGenerator:
    """
    Generates training data according to curriculum schedule.
    """
    
    def __init__(
        self,
        scheduler: CurriculumScheduler,
        model_type: str,
        batch_size: int,
        base_n_dims: int,
        nn_hidden_dim: Optional[int] = None,
        nn_output_dim: Optional[int] = None,
        nn_input_dims: Optional[int] = None
    ):
        """
        Initialize curriculum data generator.
        
        Args:
            scheduler: CurriculumScheduler instance
            model_type: "simple_regression" or "nn"
            batch_size: Batch size for data generation
            base_n_dims: Base dimensionality (used if scheduler doesn't specify)
            nn_hidden_dim: Hidden dimension for NN task
            nn_output_dim: Output dimension for NN task
            nn_input_dims: Input dimension for NN task
        """
        self.scheduler = scheduler
        self.model_type = model_type
        self.batch_size = batch_size
        self.base_n_dims = base_n_dims
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.nn_input_dims = nn_input_dims
        
        self.current_step = 0
        self.current_params = None
    
    def generate(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a batch of data according to curriculum at given step.
        
        Args:
            step: Current training step
            
        Returns:
            Tuple of (xs, ys) tensors
        """
        self.current_step = step
        self.current_params = self.scheduler.get_curriculum_params(step)
        
        n_points = self.current_params['n_points']
        noise_scale = self.current_params['noise_scale']
        n_dims = self.current_params.get('n_dims', self.base_n_dims)
        
        if self.model_type == "simple_regression":
            xs, ys = self._generate_linear_with_curriculum(
                n_points, n_dims, noise_scale
            )
        elif self.model_type == "nn":
            xs, ys = self._generate_nn_with_curriculum(
                n_points, noise_scale
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return xs, ys
    
    def _generate_linear_with_curriculum(
        self,
        n_points: int,
        n_dims: int,
        noise_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate linear regression data with curriculum parameters."""
        xs = torch.randn(self.batch_size, n_points, n_dims)
        w = torch.randn(self.batch_size, n_dims, 1)
        ys = xs @ w + noise_scale * torch.randn(self.batch_size, n_points, 1)
        return xs, ys
    
    def _generate_nn_with_curriculum(
        self,
        n_points: int,
        noise_scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate NN task data with curriculum parameters."""
        xs = torch.randn(self.batch_size, n_points, self.nn_input_dims)
        ys = []
        
        for i in range(self.batch_size):
            x = xs[i]
            hidden_layer = torch.nn.Linear(self.nn_input_dims, self.nn_hidden_dim)
            output_layer = torch.nn.Linear(self.nn_hidden_dim, self.nn_output_dim)
            
            with torch.no_grad():
                pass1 = hidden_layer(x)
                activated_pass1 = torch.nn.functional.relu(pass1)
                pass2 = output_layer(activated_pass1)
                # Add curriculum-controlled noise
                pass2 = pass2 + noise_scale * torch.randn_like(pass2)
            
            ys.append(pass2)
        
        ys_out = torch.stack(ys)
        return xs, ys_out
    
    def get_current_stage_info(self) -> str:
        """Get readable description of current curriculum stage."""
        if self.current_params is None:
            return "Not started"
        
        stage = self.current_params['stage']
        n_points = self.current_params['n_points']
        noise = self.current_params['noise_scale']
        
        info = f"Stage {stage}/{self.scheduler.num_stages-1}: "
        info += f"{n_points} points, noise={noise:.3f}"
        
        if 'n_dims' in self.current_params:
            info += f", dims={self.current_params['n_dims']}"
        
        return info


def train_with_curriculum(
    model,
    model_type: str,
    scheduler: CurriculumScheduler,
    batch_size: int,
    base_n_dims: int,
    lr: float = 1e-4,
    log_every: int = 50,
    device: str = "cpu",
    nn_hidden_dim: Optional[int] = None,
    nn_output_dim: Optional[int] = None,
    nn_input_dims: Optional[int] = None
) -> Dict:
    """
    Train a model using curriculum learning.
    
    Args:
        model: Model to train
        model_type: "simple_regression" or "nn"
        scheduler: CurriculumScheduler instance
        batch_size: Batch size
        base_n_dims: Base dimensionality
        lr: Learning rate
        log_every: Log frequency
        device: Device to train on
        nn_hidden_dim: Hidden dim for NN (if model_type == "nn")
        nn_output_dim: Output dim for NN (if model_type == "nn")
        nn_input_dims: Input dim for NN (if model_type == "nn")
    
    Returns:
        Training history dictionary
    """
    from train_NN import train_step
    
    # Initialize data generator
    data_gen = CurriculumDataGenerator(
        scheduler=scheduler,
        model_type=model_type,
        batch_size=batch_size,
        base_n_dims=base_n_dims,
        nn_hidden_dim=nn_hidden_dim,
        nn_output_dim=nn_output_dim,
        nn_input_dims=nn_input_dims
    )
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'losses': [],
        'curriculum_stages': [],
        'curriculum_params': []
    }
    
    print(f"\nStarting curriculum learning training...")
    print(f"{'='*70}\n")
    
    current_stage = -1
    
    for step in range(scheduler.total_steps):
        # Generate curriculum data
        xs, ys = data_gen.generate(step)
        xs, ys = xs.to(device), ys.to(device)
        
        # Training step
        loss = train_step(model, xs, ys, optimizer)
        history['losses'].append(loss)
        
        # Track curriculum progression
        params = data_gen.current_params
        history['curriculum_stages'].append(params['stage'])
        history['curriculum_params'].append(params)
        
        # Log progress and stage transitions
        if params['stage'] != current_stage:
            current_stage = params['stage']
            print(f"\n{'='*70}")
            print(f"Curriculum Stage Change at Step {step}")
            print(f"{data_gen.get_current_stage_info()}")
            print(f"{'='*70}\n")
        
        if step % log_every == 0:
            print(f"Step {step:5d} | Loss: {loss:.6f} | {data_gen.get_current_stage_info()}")
    
    print(f"\nCurriculum training complete!")
    return history


# Example usage
if __name__ == "__main__":
    from model import TransformerModel
    from config import n_dims, batch_size
    
    # Create a simple curriculum
    scheduler = CurriculumScheduler(
        total_steps=200000,
        num_stages=10,
        start_points=3,      # Start easy with 5 examples
        end_points=20,       # End with full 20 examples
        start_noise=0.01,    # Start with low noise
        end_noise=0.01        # End with normal noise
    )
    
    # Create model
    model = TransformerModel(n_dims, 20, name="simple_regression")
    
    # Train with curriculum
    history = train_with_curriculum(
        model=model,
        model_type="simple_regression",
        scheduler=scheduler,
        batch_size=batch_size,
        base_n_dims=n_dims,
        lr=1e-4,
        log_every=50
    )
    
    print(f"\nTraining completed!")
    print(f"Final loss: {history['losses'][-1]:.6f}")
