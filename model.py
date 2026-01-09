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


    # Replace the forward method (lines 30-44):

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys)
        B, seq_len, D = zs.shape
        
        # Create causal attention mask to prevent peeking at future positions
        # attention_mask: 1 = attend, 0 = ignore
        # Shape: (B, seq_len)
        attention_mask = torch.ones(B, seq_len, device=zs.device, dtype=torch.long)
        
        embeds = self._read_in(zs)
        output = self._backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        ).last_hidden_state
        prediction = self._read_out(output)
        return prediction


    # Replace the _combine method (lines 46-84):

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleaves x's and y's to create: x_0, y_0, x_1, y_1, ..., x_{N-2}, y_{N-2}, x_{N-1}, 0
        
        This creates n_points-1 complete (x,y) pairs, then a final x followed by 0.
        The model sees N-1 examples and predicts the final y at position 2N-1.
        
        Args:
            xs_b: (B, N, input_dim) - input features
            ys_b: (B, N, output_dim) - target values
        
        Returns:
            (B, 2N, D) - interleaved sequence where D = max(input_dim, output_dim)
        """
        B, N, input_dim = xs_b.shape
        output_dim = ys_b.shape[-1]
        D = max(input_dim, output_dim)
        
        # Pad xs to match max dimension if needed
        if input_dim < D:
            xs_b = F.pad(xs_b, (0, D - input_dim))
        
        # Create y sequence: [y_0, y_1, ..., y_{N-2}, 0]
        # All actual y values except the last one (which we predict), plus 0 at the end
        ys_in = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)
        ys_in[:, :-1, :output_dim] = ys_b[:, :-1, :]  # Copy all y's except the last
        
        # Interleave: x_0, y_0, x_1, y_1, ..., x_{N-1}, 0
        toks = []
        for i in range(N):
            toks.append(xs_b[:, i, :])   # (B, D)  x_i
            toks.append(ys_in[:, i, :])  # (B, D)  y_i for i<N-1, else 0
        
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


    # Replace the forward method (lines 112-126):

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        
        zs = self._combine(xs, ys)
        B, seq_len, D = zs.shape
        
        # Create causal attention mask to prevent peeking at future positions
        # attention_mask: 1 = attend, 0 = ignore
        # Shape: (B, seq_len)
        attention_mask = torch.ones(B, seq_len, device=zs.device, dtype=torch.long)
        
        embeds = self._read_in(zs)
        output = self._backbone(
            inputs_embeds=embeds,
            attention_mask=attention_mask
        ).last_hidden_state
        prediction = self._read_out(output)
        return prediction


    # Replace the _combine method (lines 128-177):

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleaves x's and y's to create: x_0, y_0, x_1, y_1, ..., x_{N-2}, y_{N-2}, x_{N-1}, 0
        
        This creates n_points-1 complete (x,y) pairs, then a final x followed by 0.
        The model sees N-1 examples and predicts the final y at position 2N-1.
        
        Args:
            xs_b: (B, N, input_dim) - input features
            ys_b: (B, N, output_dim) - target values
        
        Returns:
            (B, 2N, D) - interleaved sequence where D = max(input_dim, output_dim)
        """
        B, N, input_dim = xs_b.shape
        output_dim = ys_b.shape[-1]
        D = max(input_dim, output_dim)
        
        # Pad xs to match max dimension if needed
        if input_dim < D:
            xs_b = F.pad(xs_b, (0, D - input_dim))
        
        # Create y sequence: [y_0, y_1, ..., y_{N-2}, 0]
        # All actual y values except the last one (which we predict), plus 0 at the end
        ys_in = torch.zeros(B, N, D, device=xs_b.device, dtype=xs_b.dtype)
        ys_in[:, :-1, :output_dim] = ys_b[:, :-1, :]  # Copy all y's except the last
        
        # Interleave: x_0, y_0, x_1, y_1, ..., x_{N-1}, 0
        toks = []
        for i in range(N):
            toks.append(xs_b[:, i, :])   # (B, D)  x_i
            toks.append(ys_in[:, i, :])  # (B, D)  y_i for i<N-1, else 0
        
        return torch.stack(toks, dim=1)  # (B, 2N, D)




