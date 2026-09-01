import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_candidate = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, h_prev):
        gates = self.conv_gates(torch.cat([x, h_prev], dim=1))
        reset, update = gates.chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.conv_candidate(torch.cat([x, reset * h_prev], dim=1)))
        return (1 - update) * h_prev + update * candidate

    def init_hidden(self, batch_size, height, width):
        return torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=self.conv_gates.weight.device,
        )


class ConvGRU(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, kernel_size=3, bias=True):
        super().__init__()
        cells = []
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else hidden_channels
            cells.append(ConvGRUCell(layer_in, hidden_channels, kernel_size, bias))
        self.layers = nn.ModuleList(cells)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _, height, width = x.shape
        if hidden is None:
            hidden_states = [cell.init_hidden(batch_size, height, width) for cell in self.layers]
        else:
            hidden_states = list(hidden)

        outputs = []
        for t in range(seq_len):
            current = x[:, t]
            for i, cell in enumerate(self.layers):
                current = cell(current, hidden_states[i])
                hidden_states[i] = current
            outputs.append(current)
        return torch.stack(outputs, dim=1), hidden_states


class NextFramePredictor(nn.Module):
    def __init__(self, latent_channels, hidden_channels, num_layers, kernel_size=3):
        super().__init__()
        self.latent_channels = latent_channels
        self.gru = ConvGRU(self.latent_channels * 2, hidden_channels, num_layers, kernel_size)
        self.head = nn.Conv2d(hidden_channels, self.latent_channels, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _predict_from_pairs(self, pairs, hidden=None):
        out, hidden = self.gru(pairs, hidden)
        b, t, h_ch, h, w = out.shape
        delta = self.head(out.reshape(b * t, h_ch, h, w)).view(b, t, self.latent_channels, h, w)
        r_curr = pairs[:, :, self.latent_channels :]
        return F.relu(r_curr + delta), hidden

    def init_hidden_from_context(self, r_ctx):
        if r_ctx.shape[1] < 2:
            raise ValueError("burn-in needs at least 2 context frames")
        if r_ctx.shape[1] == 2:
            return r_ctx[:, 0], r_ctx[:, 1], None
        pairs = torch.cat([r_ctx[:, :-2], r_ctx[:, 1:-1]], dim=2)
        _, hidden = self._predict_from_pairs(pairs)
        return r_ctx[:, -2], r_ctx[:, -1], hidden

    def _burnin(self, r_ctx):
        return self.init_hidden_from_context(r_ctx)

    def predict_latent(self, r_seq, teacher_force=True, context=2):
        if teacher_force:
            pairs = torch.cat([r_seq[:, :-2], r_seq[:, 1:-1]], dim=2)
            preds, _ = self._predict_from_pairs(pairs)
            return preds
        return self.rollout(r_seq[:, :context], n_steps=r_seq.shape[1] - context)

    def rollout(self, r_ctx, n_steps, hidden=None):
        if hidden is None:
            r_prev, r_curr, hidden = self.init_hidden_from_context(r_ctx)
        else:
            r_prev, r_curr = r_ctx[:, -2], r_ctx[:, -1]
        preds = []
        for _ in range(n_steps):
            pair = torch.cat([r_prev, r_curr], dim=1).unsqueeze(1)
            r_next, hidden = self._predict_from_pairs(pair, hidden)
            r_next = r_next[:, 0]
            preds.append(r_next)
            r_prev, r_curr = r_curr, r_next
        if not preds:
            return r_ctx.new_zeros(
                r_ctx.shape[0], 0, self.latent_channels, r_ctx.shape[-2], r_ctx.shape[-1]
            )
        return torch.stack(preds, dim=1)
