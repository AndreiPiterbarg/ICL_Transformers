import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from config import batch_size, nn_output_dim


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, name, n_embd=128, n_layer=6, n_head=4):
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
        self.name = name

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


        B, N, input_dim = xs_b.shape

        output_dim = ys_b.shape[-1]

 

        D = max(input_dim, output_dim)

        if input_dim < D:

            xs_b = F.pad(xs_b, (0, D - input_dim))  # pad last dim

 

        # Create shifted y sequence: [0, y_0, y_1, ..., y_{N-2}]

        ys_in = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)

        # Shift y values by one position: position i gets y_{i-1}

        ys_in[:, 1:, :output_dim] = ys_b[:, :-1, :]  # ys_in[i] = y[i-1] for i >= 1, else 0

 

        toks = []

        for i in range(N):

            toks.append(xs_b[:, i, :])   # (B, D)  x_i

            toks.append(ys_in[:, i, :])  # (B, D)  y_{i-1} (or 0 for i=0)

 

        return torch.stack(toks, dim=1)  # (B, 2N, D)



class NNTransformer(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, n_positions, name, n_embd=128, n_layer=6, n_head=4):
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
        self.n_input_dims = n_input_dims
        self.n_output_dims = n_output_dims
        max_dim = max(n_input_dims, n_output_dims)
        self.n_positions = n_positions
        self._read_in = nn.Linear(max_dim, n_embd)
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

        """Interleaves the x's and the y's into a single sequence with proper causal masking.

 

        The sequence is constructed so the model cannot see the target y when predicting it:

        x_0, 0, x_1, y_0, x_2, y_1, ..., x_{N-1}, y_{N-2}

 

        This ensures at position 2i+1 the model predicts y_i seeing only x_0,...,x_i,y_0,...,y_{i-1}

        """

        B, N, input_dim = xs_b.shape

        output_dim = ys_b.shape[-1]

 

        D = max(input_dim, output_dim)

        if input_dim < D:

            xs_b = F.pad(xs_b, (0, D - input_dim))      # pad last dim

 

        # Create shifted y sequence: [0, y_0, y_1, ..., y_{N-2}]

        ys_in = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)

        # Shift y values by one position: position i gets y_{i-1}

        ys_in[:, 1:, :output_dim] = ys_b[:, :-1, :]  # ys_in[i] = y[i-1] for i >= 1, else 0

 

        toks = []

        for i in range(N):

            toks.append(xs_b[:, i, :])   # (B, D)  x_i

            toks.append(ys_in[:, i, :])  # (B, D)  y_{i-1} (or 0 for i=0)

        return torch.stack(toks, dim=1)  # (B, 2N, D)



