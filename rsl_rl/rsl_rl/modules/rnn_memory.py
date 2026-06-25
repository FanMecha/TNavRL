import torch
import torch.nn as nn
from typing import Optional, Tuple
from rsl_rl.utils import unpad_trajectories


class LSTMSRUCell(nn.Module):
    """LSTM cell with transformation gate and polynomial refinement.

    This cell uses a transformation gate and a refined forget gate based on:
    https://proceedings.mlr.press/v119/gu20a/gu20a.pdf
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Combined linear layer for all gates
        self.linear_all = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)

        # initialize all weights to orthogonal
        nn.init.orthogonal_(self.linear_all.weight)

        # Initialize forgetting gate bias to 1 (+ delta to break symmetry)
        if bias:
            self.linear_all.bias.data[hidden_size : 2 * hidden_size] = 1.0 + torch.randn(hidden_size)

        # Transformation Gate
        self.transform_gate = nn.Linear(input_size, hidden_size, bias=bias)

        # Initialize transform gate weights to orthogonal
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute all gates in a single linear transformation, and transform gate
        gates = self.linear_all(combined)
        tx = self.transform_gate(x)

        # Split gates into input, forget, cell, and output
        i, f, o, g = torch.split(gates, self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g_t = torch.tanh(tx * g)

        # Refine Gate: https://proceedings.mlr.press/v119/gu20a/gu20a.pdf
        f = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f**2
        # Compute the new cell state
        c_next = f * c + (1.0 - f) * g_t

        # Compute the new hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTM_SRU(nn.Module):
    """Multi-layer LSTM with SRU-style gating.

    Args:
        input_size: The number of expected features in the input.
        hidden_size: The number of features in the hidden state.
        num_layers: Number of recurrent layers. Default: 1.
        batch_first: If True, input/output tensors are (batch, seq, feature).
            Default: False.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList(
            [LSTMSRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, x, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert batch_first to time_first

        seq_len, batch_size, _ = x.shape

        if state is None:
            state = self.init_state(batch_size, x.device)

        # Pre-allocate output tensor to avoid per-timestep torch.stack()
        outputs = torch.empty(seq_len, batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        h, c = state
        for t in range(seq_len):
            x_t = x[t]
            new_h, new_c = [], []
            for layer_idx, cell in enumerate(self.cells):
                h_t, c_t = cell(x_t, h[layer_idx], c[layer_idx])
                new_h.append(h_t)
                new_c.append(c_t)
                x_t = h_t  # Output of current layer is input to the next
            h = torch.stack(new_h)
            c = torch.stack(new_c)
            outputs[t] = h[-1]
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # Convert time_first back to batch_first

        return outputs, (h, c)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h_0, c_0)


class MemorySRU(torch.nn.Module):
    """Memory module using plain LSTM_SRU (without attention).

    Attention is applied externally in batched mode for efficiency.

    Args:
        input_size: Size of input features (after attention + proprioceptive).
        num_layers: Number of RNN layers.
        hidden_size: Hidden state size.
    """

    def __init__(self, input_size: int, num_layers: int = 1, hidden_size: int = 256):
        super().__init__()
        print(f"[MemorySRU] Init: input_size={input_size}, num_layers={num_layers}, hidden_size={hidden_size}")

        self.rnn = LSTM_SRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.hidden_states = None

    def forward(self, input_features, masks=None, hidden_states=None):
        """Forward pass through the memory module.

        Args:
            input_features: Combined features tensor (after attention).
            masks: Mask tensor for batch mode (None for inference).
            hidden_states: Hidden states from previous step.

        Returns:
            RNN output tensor.
        """
        batch_mode = masks is not None
        if batch_mode:
            # In batch mode, hidden states must be provided.
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input_features, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # In inference mode, use (or initialize) the module's hidden state.
            out, self.hidden_states = self.rnn(input_features.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones, use_random_init=True):
        """Reset hidden states for terminated episodes.

        Memory-optimized: only operates on terminated environments instead of
        cloning the entire hidden state tensor.
        """
        if self.hidden_states is None:
            return

        done_mask = dones == 1
        if not done_mask.any():
            return

        done_indices = done_mask.nonzero(as_tuple=True)[0]
        num_done = done_indices.size(0)
        batch_size = dones.size(0)

        for hidden_state in self.hidden_states:
            if use_random_init:
                # Sample random indices from entire batch (preserves original semantics)
                random_indices = torch.randint(0, batch_size, (num_done,), device=hidden_state.device)
                # Clone only the selected states to avoid in-place modification issues
                hidden_state[..., done_indices, :] = hidden_state[..., random_indices, :].clone()
            else:
                # Use zero initialization
                hidden_state[..., done_indices, :] = 0.0