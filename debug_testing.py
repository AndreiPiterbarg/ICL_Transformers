"""

Comprehensive diagnostic test for ICL transformer

Tests EVERYTHING that could be wrong

"""

import torch

import torch.nn.functional as F

import numpy as np

from model import TransformerModel

from data_sampler import generate_linear

from config import n_dims, n_points, batch_size, lr

 

print("="*80)

print("COMPREHENSIVE ICL TRANSFORMER DIAGNOSTIC")

print("="*80)

 

# ============================================================================

# TEST 1: Data Generation Sanity Check

# ============================================================================

print("\n" + "="*80)

print("TEST 1: DATA GENERATION")

print("="*80)

 

xs, ys = generate_linear(n_points, batch_size, n_dims)

print(f"Data shapes: xs={xs.shape}, ys={ys.shape}")

print(f"Config: n_points={n_points}, batch_size={batch_size}, n_dims={n_dims}")

 

# Check that y actually depends on x (verify linear relationship)

# For each batch, compute correlation between x and y

correlations = []

for b in range(min(5, batch_size)):

    X = xs[b].numpy()  # (N, D)

    Y = ys[b].numpy()  # (N, 1)

 

    # Fit least squares

    w_ls = np.linalg.lstsq(X, Y, rcond=None)[0]

    Y_pred = X @ w_ls

 

    mse = np.mean((Y - Y_pred)**2)

    y_var = np.var(Y)

    r2 = 1 - mse / y_var

    correlations.append(r2)

 

print(f"R² scores for first 5 batches: {[f'{r:.4f}' for r in correlations]}")

print(f"Average R²: {np.mean(correlations):.4f}")

if np.mean(correlations) > 0.9:

    print("✓ Data generation looks correct (strong linear relationship)")

else:

    print("✗ WARNING: Weak linear relationship in data!")

 

# Check noise level

print(f"\nData statistics:")

print(f"  xs mean: {xs.mean():.4f}, std: {xs.std():.4f}")

print(f"  ys mean: {ys.mean():.4f}, std: {ys.std():.4f}")

 

# ============================================================================

# TEST 2: Sequence Construction

# ============================================================================

print("\n" + "="*80)

print("TEST 2: SEQUENCE CONSTRUCTION (_combine method)")

print("="*80)

 

model = TransformerModel(n_dims, n_points, name="simple_regression")

zs = model._combine(xs, ys)

 

print(f"Combined sequence shape: {zs.shape}")

print(f"Expected: ({batch_size}, {2*n_points}, {n_dims})")

 

if zs.shape[1] != 2 * n_points:

    print(f"✗ ERROR: Wrong sequence length! Got {zs.shape[1]}, expected {2*n_points}")

else:

    print(f"✓ Sequence length correct")

 

# Check the structure for first batch

print(f"\nChecking sequence structure (first batch, first 6 positions):")

print("Expected: x_0, [0,...], x_1, [y_0,...], x_2, [y_1,...]")

 

for i in range(min(6, zs.shape[1])):

    pos_type = "x" if i % 2 == 0 else "y"

    if pos_type == "x":

        expected_idx = i // 2

        actual = zs[0, i, :]

        expected = xs[0, expected_idx, :]

        match = torch.allclose(actual, expected, atol=1e-5)

        print(f"  Pos {i} (x_{expected_idx}): Match={match}")

    else:

        expected_idx = i // 2 - 1  # Should contain y_{i//2 - 1}

        actual = zs[0, i, 0]  # y is in first dimension

        if expected_idx < 0:

            expected = 0.0

        else:

            expected = ys[0, expected_idx, 0]

        match = abs(actual.item() - expected.item()) < 1e-5 if isinstance(expected, torch.Tensor) else abs(actual.item() - expected) < 1e-5

        print(f"  Pos {i} (y_{expected_idx if expected_idx >= 0 else 'none'}): actual={actual.item():.4f}, expected={expected if isinstance(expected, float) else expected.item():.4f}, Match={match}")

 

# Critical check: At position 2i+1, verify we DON'T see y_i (data leakage check)

print(f"\n*** DATA LEAKAGE CHECK ***")

print("At position 2i+1, we should see y_(i-1), NOT y_i")

for i in range(min(3, n_points)):

    pos = 2*i + 1

    y_at_pos = zs[0, pos, 0].item()

 

    if i == 0:

        expected = 0.0

        actual_y_idx = "none"

    else:

        expected = ys[0, i-1, 0].item()

        actual_y_idx = i-1

 

    leak_value = ys[0, i, 0].item()  # This is what we're predicting

 

    is_leak = abs(y_at_pos - leak_value) < 1e-5

    is_correct = abs(y_at_pos - expected) < 1e-5 if isinstance(expected, float) else abs(y_at_pos - expected.item()) < 1e-5

 

    print(f"  Position {pos} (predicting y_{i}):")

    print(f"    Input contains: {y_at_pos:.4f}")

    print(f"    Should contain y_{actual_y_idx}: {expected if isinstance(expected, float) else expected:.4f}")

    print(f"    Target y_{i}: {leak_value:.4f}")

    print(f"    Is leak? {is_leak} | Is correct? {is_correct}")

 

    if is_leak:

        print(f"    ✗✗✗ DATA LEAK DETECTED! ✗✗✗")

    elif is_correct:

        print(f"    ✓ No leak")

 

# ============================================================================

# TEST 3: Model Forward Pass

# ============================================================================

print("\n" + "="*80)

print("TEST 3: MODEL FORWARD PASS")

print("="*80)

 

model.eval()

with torch.no_grad():

    preds = model(xs, ys)

 

print(f"Predictions shape: {preds.shape}")

print(f"Expected: ({batch_size}, {2*n_points}, 1)")

 

if preds.shape[1] != 2 * n_points:

    print(f"✗ ERROR: Wrong predictions length! Got {preds.shape[1]}, expected {2*n_points}")

else:

    print(f"✓ Predictions length correct")

 

# ============================================================================

# TEST 4: Training Supervision Positions

# ============================================================================

print("\n" + "="*80)

print("TEST 4: TRAINING SUPERVISION ALIGNMENT")

print("="*80)

 

# This mimics what train_step does

B, N, _ = ys.shape

y_pos = torch.arange(1, 2 * N, 2)

 

print(f"N = {N}")

print(f"Supervision positions (y_pos): {y_pos.tolist()[:10]}... (first 10)")

print(f"Total supervision positions: {len(y_pos)}")

print(f"Should be: {N}")

 

if len(y_pos) != N:

    print(f"✗ ERROR: Wrong number of supervision positions!")

else:

    print(f"✓ Correct number of supervision positions")

 

# Check if all positions are in bounds

max_pos = y_pos.max().item()

print(f"Max supervision position: {max_pos}")

print(f"Predictions available up to position: {preds.shape[1] - 1}")

 

if max_pos >= preds.shape[1]:

    print(f"✗✗✗ CRITICAL ERROR: Supervision position {max_pos} out of bounds! ✗✗✗")

    print(f"This will cause index errors!")

else:

    print(f"✓ All supervision positions in bounds")

 

# Extract predictions at y positions

pred_at_y = preds[:, y_pos, :]

print(f"\nPredictions at y positions shape: {pred_at_y.shape}")

print(f"Targets shape: {ys.shape}")

print(f"Shapes match? {pred_at_y.shape == ys.shape}")

 

# ============================================================================

# TEST 5: What Does Model See at Each Prediction Position?

# ============================================================================

print("\n" + "="*80)

print("TEST 5: CAUSAL ATTENTION CONTEXT")

print("="*80)

 

print("At each position where we predict y_i, what has the model seen?")

print("(GPT2 uses causal attention: at position p, can see positions 0 to p)\n")

 

for i in range(min(5, N)):

    pred_pos = y_pos[i].item()

    print(f"Predicting y_{i} at position {pred_pos}:")

    print(f"  Model sees positions: 0 to {pred_pos}")

 

    # Count what it's seen

    n_x_seen = (pred_pos + 1 + 1) // 2  # x positions: 0, 2, 4, ..., up to pred_pos

    n_y_seen = pred_pos // 2  # y positions: 1, 3, 5, ..., up to pred_pos-1 or pred_pos-2

 

    print(f"  That's {n_x_seen} x values: x_0 to x_{n_x_seen-1}")

    print(f"  And {n_y_seen} y values: y_0 to y_{n_y_seen-1}")

    print(f"  Expected: {i+1} x values, {i} y values")

 

    if n_x_seen == i+1 and n_y_seen == i:

        print(f"  ✓ Correct context!")

    else:

        print(f"  ✗ ERROR: Wrong context!")

 

# ============================================================================

# TEST 6: Baseline Least Squares

# ============================================================================

print("\n" + "="*80)

print("TEST 6: BASELINE LEAST SQUARES PERFORMANCE")

print("="*80)

 

def compute_ls_baseline(xs, ys):

    B, N, D = xs.shape

    xs_context = xs[:, :-1, :]

    ys_context = ys[:, :-1, :]

    xs_query = xs[:, -1:, :]

    ys_query = ys[:, -1:, :]

 

    query_losses = []

    for b in range(B):

        X = xs_context[b]

        Y = ys_context[b].squeeze(-1)

 

        try:

            XtX = X.T @ X

            XtX_reg = XtX + 1e-5 * torch.eye(D)

            w_ls = torch.linalg.solve(XtX_reg, X.T @ Y)

            y_pred = xs_query[b] @ w_ls.unsqueeze(-1)

            loss = F.mse_loss(y_pred, ys_query[b])

            query_losses.append(loss.item())

        except:

            query_losses.append(float('inf'))

 

    return query_losses

 

baseline_losses = compute_ls_baseline(xs, ys)

print(f"Baseline query losses (first 5): {[f'{l:.6f}' for l in baseline_losses[:5]]}")

print(f"Mean baseline loss: {np.mean(baseline_losses):.6f}")

print(f"Std baseline loss: {np.std(baseline_losses):.6f}")

 

if np.mean(baseline_losses) < 0.1:

    print("✓ Baseline achieves good performance")

else:

    print("✗ WARNING: Baseline performance is poor!")

 

# ============================================================================

# TEST 7: Untrained Model Performance

# ============================================================================

print("\n" + "="*80)

print("TEST 7: UNTRAINED MODEL PERFORMANCE")

print("="*80)

 

model.eval()

with torch.no_grad():

    preds = model(xs, ys)

    pred_at_y = preds[:, y_pos, :]

 

    # Query loss (last position)

    query_loss = F.mse_loss(pred_at_y[:, -1, :], ys[:, -1, :])

 

    # All positions loss

    total_loss = F.mse_loss(pred_at_y, ys)

 

print(f"Untrained model query loss: {query_loss.item():.6f}")

print(f"Untrained model total loss: {total_loss.item():.6f}")

print(f"Baseline query loss: {np.mean(baseline_losses):.6f}")

print(f"Ratio (model/baseline): {query_loss.item()/np.mean(baseline_losses):.2f}x")

 

print(f"\nRandom prediction baseline (predicting 0):")

random_baseline = F.mse_loss(torch.zeros_like(ys[:, -1, :]), ys[:, -1, :])

print(f"  Loss: {random_baseline.item():.6f}")

print(f"  Model vs random: {query_loss.item()/random_baseline.item():.2f}x")

 

if query_loss.item() > random_baseline.item() * 0.9:

    print("✗ Model is performing like random guessing")

else:

    print("✓ Model is better than random")

 

# ============================================================================

# TEST 8: Training Step Simulation

# ============================================================================

print("\n" + "="*80)

print("TEST 8: TRAINING STEP SIMULATION")

print("="*80)

 

model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

 

print(f"Learning rate: {lr}")

print(f"Running 10 training steps...")

 

losses = []

for step in range(10):

    xs_train, ys_train = generate_linear(n_points, batch_size, n_dims)

 

    optimizer.zero_grad()

    preds = model(xs_train, ys_train)

 

    B, N, _ = ys_train.shape

    y_pos = torch.arange(1, 2 * N, 2)

    pred_at_y = preds[:, y_pos, :]

 

    loss = F.mse_loss(pred_at_y, ys_train)

    loss.backward()

 

    # Check gradients

    total_grad_norm = 0.0

    n_params = 0

    for p in model.parameters():

        if p.grad is not None:

            total_grad_norm += p.grad.norm().item() ** 2

            n_params += 1

    total_grad_norm = total_grad_norm ** 0.5

 

    optimizer.step()

    losses.append(loss.item())

 

    if step == 0:

        print(f"  Step {step}: loss={loss.item():.6f}, grad_norm={total_grad_norm:.6f}, n_params_with_grad={n_params}")

 

print(f"\nLosses: {[f'{l:.4f}' for l in losses]}")

print(f"First loss: {losses[0]:.6f}")

print(f"Last loss: {losses[-1]:.6f}")

print(f"Change: {losses[-1] - losses[0]:.6f}")

 

if total_grad_norm < 1e-6:

    print("✗ ERROR: Gradients are too small!")

elif total_grad_norm > 1e3:

    print("✗ WARNING: Gradients are very large!")

else:

    print("✓ Gradient norms look reasonable")

 

if abs(losses[-1] - losses[0]) < 0.01:

    print("✗ WARNING: Loss barely changed during training")

elif losses[-1] > losses[0]:

    print("✗ WARNING: Loss increased during training")

else:

    print("✓ Loss is decreasing")

 

# ============================================================================

# TEST 9: Model Capacity Check

# ============================================================================

print("\n" + "="*80)

print("TEST 9: MODEL CAPACITY")

print("="*80)

 

n_params = sum(p.numel() for p in model.parameters())

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

 

print(f"Total parameters: {n_params:,}")

print(f"Trainable parameters: {n_trainable:,}")

print(f"Model size: ~{n_params * 4 / 1e6:.2f} MB (assuming float32)")

 

print(f"\nModel architecture:")

print(f"  n_embd: 128")

print(f"  n_layer: 6")

print(f"  n_head: 4")

print(f"  n_positions: {2*n_points}")

 

# ============================================================================

# TEST 10: Evaluation Code Check

# ============================================================================

print("\n" + "="*80)

print("TEST 10: EVALUATION CODE ALIGNMENT")

print("="*80)

 

# Make sure eval uses same positions as training

print("Checking that evaluation uses same logic as training...")

 

xs_eval, ys_eval = generate_linear(n_points, batch_size, n_dims)

 

model.eval()

with torch.no_grad():

    preds_eval = model(xs_eval, ys_eval)

 

    B, N, _ = ys_eval.shape

    y_pos_eval = torch.arange(1, 2 * N, 2)

    pred_at_y_eval = preds_eval[:, y_pos_eval, :]

 

    eval_query_loss = F.mse_loss(pred_at_y_eval[:, -1, :], ys_eval[:, -1, :])

 

print(f"Eval query loss: {eval_query_loss.item():.6f}")

print("✓ Evaluation code uses same position indexing as training")

 

# ============================================================================

# SUMMARY

# ============================================================================

print("\n" + "="*80)

print("DIAGNOSTIC SUMMARY")

print("="*80)

print("\nRun this test and send me ALL the output.")

print("I'll analyze it to find exactly what's wrong.")

print("="*80)