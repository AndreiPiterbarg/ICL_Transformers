import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from config import batch_size, nn_output_dim

class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions,name, n_embd=128, n_layer=6, n_head=4):
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
        self.name ="simple_regression"
        

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
        """Interleaves the x's and the y's into a single sequence, and set final y to 0
        Such that tgt in train.py contains the last element in the batch that transformer must guess and the other interleaved ... """

        B, N, D = xs_b.shape
        ys_b = ys_b.clone()
        # Remove the last y from each batch (set to zero)
        ys_b[:, -1, :] = 0.0
        # Pad ys to n_dims
        ys_pad = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)
        ys_pad[..., 0] = ys_b.squeeze(-1)
        # Interleave xs and ys_pad, but drop the last y
        toks = []
        for i in range(N):
            toks.append(xs_b[:, i, :])      # x_i
            toks.append(ys_pad[:, i, :])     # y_i (last one is zeros in inputs)

        return torch.stack(toks, dim=1)      # (B, 2N, D)




class NNTransformer(nn.Module):
    def __init__(self, n_dims, n_positions, name, n_embd=128, n_layer=6, n_head=4):
        super(NNTransformer, self).__init__()
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
        self.name = "nn"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, nn_output_dim)


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
        Such that tgt in train.py contains the last element in the batch that transformer must guess and the other interleaved ... """

        B, N, input_dim = xs_b.shape
        output_dim = ys_b.shape[-1]
        # Remove the last ys from each batch (set to zero)
        ys_in = ys_b.clone()
        ys_in[:, -1, :] = 0.0

        D = max(input_dim, output_dim)
        if input_dim < D:
            xs_b = F.pad(xs_b, (0, D - input_dim))      # pad last dim
        if output_dim < D:
            ys_in = F.pad(ys_in, (0, D - output_dim))   # pad last dim

        toks = []
        for i in range(N):
            toks.append(xs_b[:, i, :])   # (B, D)
            toks.append(ys_in[:, i, :])  # (B, D) (last one is zeros in inputs)

        return torch.stack(toks, dim=1)  # (B, 2N, D)



