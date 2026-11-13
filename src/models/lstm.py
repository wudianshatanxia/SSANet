from __future__ import annotations

import torch
from torch import nn


class LSTMScoreNet(nn.Module):
    """Score users with a two-step temporal LSTM."""

    def __init__(self, pmi_real_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pmi_real_dim = pmi_real_dim
        self.step_dim = pmi_real_dim + 3
        self.lstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, F = x.shape
        D = self.pmi_real_dim
        now_pmi = x[:, :, :D]
        now_rate = x[:, :, D : D + 1]
        fut_pmi = x[:, :, D + 1 : D + 1 + D]
        fut_rate = x[:, :, D + 1 + D : D + 1 + D + 1]
        hist = x[:, :, D + 1 + D + 1 : D + 1 + D + 2]
        speed = x[:, :, D + 1 + D + 2 : D + 1 + D + 3]

        step_now = torch.cat([now_pmi, now_rate, hist, speed], dim=-1)
        step_future = torch.cat([fut_pmi, fut_rate, hist, speed], dim=-1)
        seq = torch.stack([step_now, step_future], dim=2)
        seq = seq.reshape(B * K, 2, self.step_dim)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        logits = self.head(last).reshape(B, K)
        return logits
