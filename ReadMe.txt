# In-Context Learning with Transformers

 
A PyTorch implementation exploring how transformers learn to perform regression tasks in-context, inspired by "What Can Transformers Learn In-Context?" (Garg et al., 2022).

 

## What This Does

 

This project trains a GPT2-based transformer to learn regression functions purely from context - no gradient updates at test time. You give it a few (x, y) pairs from some unknown function, then a new x, and it predicts the corresponding y. The cool part? The model never explicitly learns the function during training. It learns the *algorithm* for learning.

 

## The Setup

 

The model sees a sequence structured like this:

```

x₀, y₀, x₁, y₁, x₂, y₂, ..., x_{N-2}, y_{N-2}, x_{N-1}, 0

```

 

At each training step:

- We give it **N-1 complete examples** (pairs of x and y)

- Then a final x with a zero placeholder

- The model predicts the final y at position 2N-1

- We only compute loss on this final prediction (one guess per task)

 

This forces the model to actually learn from the context, not just memorize patterns.

 


 

## Implemented Curriculum Learning

Training starts easy and gradually increases difficulty:

- **Points**: Start with 5 examples, gradually increase to N (default 10)

- **Noise**: Start with clean data (0.01), increase to realistic noise (0.1)

- **Stages**: 4 stages with 10% warmup period

 

This dramatically improves convergence. The model learns the basics on simple problems before tackling harder ones.
 

## How to Run

Simply run curriculum_learning.py or:
 

### Basic Training

```python

from model import TransformerModel

from train_NN import train

 

# Create model

model = TransformerModel(n_dims=4, n_positions=10, name="simple_regression")

 

# Train with curriculum learning (default)

losses = train(model, train_steps=200000, log_every=50, use_curriculum=True)

```

 

### Without Curriculum

```python

losses = train(model, train_steps=200000, use_curriculum=False)

```

 


## References

Based on ideas from:

- Garg et al. "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (2022)

- Brown et al. "Language Models are Few-Shot Learners" (2020)
