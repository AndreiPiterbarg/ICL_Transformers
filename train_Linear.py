import torch
import torch.nn.functional as F

def train_step(model, xs, ys, optimizer):
    """
    xs: (B, N, D_in)
    ys: (B, N, 1)     # includes y_query as ys[:, -1]
    model(xs, ys) returns preds: (B, 2N, 1) aligned with interleaved sequence
    """
    model.train()
    optimizer.zero_grad()

    preds = model(xs, ys)  # (B, 2N, 1)
    B, N, _ = ys.shape

    # x positions in the interleaved sequence: 0,2,4,...,2N-2 (length N)
    x_pos = torch.arange(0, 2 * N, 2, device=preds.device)

    # predictions at x positions -> should predict corresponding y_i
    pred_at_x = preds.index_select(dim=1, index=x_pos).squeeze(-1)  # (B, N)
    tgt_y = ys.squeeze(-1)                                          # (B, N)

    # Train loss over ALL points (including query point at i = N-1)
    loss = F.mse_loss(pred_at_x, tgt_y)

    # Useful to log query-only loss too (last point)
    query_loss = F.mse_loss(pred_at_x[:, -1], tgt_y[:, -1])

    loss.backward()
    optimizer.step()

    return loss.item(), query_loss.item()


def train(model, train_steps=1000, log_every=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(train_steps):
    #for i in range(1):


        xs, ys = generate_linear(n_points, batch_size, n_dims)


        #print (xs)
        #print (ys)
        loss = train_step(model, xs, ys, optimizer)

        if i % log_every == 0:
            print(f"step {i} | query loss: {loss:.6f}")
        #print(f"step {i} | query loss: {loss:.6f}")

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def check_gradient_flow(model, tiny=1e-10, huge=1e+2):
    stats = []
    for name, p in model.named_parameters():
        if p.grad is None:
            stats.append((name, None))
            continue
        g = p.grad.detach()
        gnorm = g.norm().item()
        stats.append((name, gnorm))

    # Print or log a compact report
    no_grad = [n for n, g in stats if g is None]
    tiny_grad = [n for n, g in stats if g is not None and g < tiny]
    huge_grad = [n for n, g in stats if g is not None and g > huge]

    print(f"[grad-flow] no_grad: {len(no_grad)} | tiny: {len(tiny_grad)} | huge: {len(huge_grad)}")
    # Uncomment for detail:
    # for n in tiny_grad: print("tiny  ", n)
    # for n in huge_grad: print("huge! ", n)

def visualize_sequence(xs, ys, max_examples=1, max_dims=4):
    """
    Show: x0, y0, x1, y1, ..., x_{N-1}, y_{N-1} (last y zero in inputs).
    """
    import torch
    B, N, D = xs.shape
    model_like_z = TransformerModel._combine(xs, ys)  # uses your combine
    for b in range(min(B, max_examples)):
        print(f"--- Batch {b} ---")
        for t in range(2*N):
            kind = "x" if (t % 2 == 0) else "y"
            vec = model_like_z[b, t, :max_dims].tolist()
            print(f"t={t:02d} [{kind}] : {vec}")
        # Also show the true y for the final slot (to confirm target)
        print("true y_N:", ys[b, -1, 0].item())

if __name__ == "__main__":
    t = TransformerModel(n_dims, n_points, name="simple_regression")
    train(t)