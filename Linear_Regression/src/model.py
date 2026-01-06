import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=6, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)


    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        
        zs = self._combine(xs, ys)

        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction
    
    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence, and remove the final y from each batch
        Such that tgt in train.py contains the last element in the batch that transformer must guess and the other interleaved ... are here"""

        B, N, D = xs_b.shape
        ys_b = ys_b.clone()
        # Remove the last y from each batch (set to zero)
        ys_b[:, -1, :] = 0.0
        # Pad ys to n_dims
        ys_pad = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)
        ys_pad[..., 0] = ys_b.squeeze(-1)
        # Interleave xs and ys_pad, but drop the last y
        zs = []
        for i in range(N):
            zs.append(xs_b[:, i, :])
            if i < N - 1:
                zs.append(ys_pad[:, i, :])
        zs = torch.stack(zs, dim=1)
        return zs