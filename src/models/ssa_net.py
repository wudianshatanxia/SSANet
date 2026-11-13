from __future__ import annotations

import torch
from torch import nn

from src.metrics import torch_complex_correlation_matrix


class SequenceShuffleAttention(nn.Module):
    """Apply feature shuffling and cross-user attention."""

    def __init__(self, embed_dim: int = 128, groups: int = 8, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.groups = groups
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 3, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _feature_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, k, d = x.shape
        groups = min(self.groups, d)
        if d % groups:
            return x
        return x.reshape(b, k, groups, d // groups).transpose(2, 3).contiguous().reshape(b, k, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self._feature_shuffle(self.norm1(x))
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.dropout(attn_out)
        return x + self.dropout(self.ffn(self.norm2(x)))


class SSANet(nn.Module):
    """Score users with calibrated predictions and Q-guided attention."""

    def __init__(
        self,
        pmi_real_dim: int,
        num_trps: int = 24,
        antennas_per_trp: int = 8,
        embed_dim: int = 128,
        dropout: float = 0.1,
        eta: float = 0.4,
        fairness_alpha: float = 0.5,
        eps: float = 1e-6,
        learned_correction_scale: float = 0.70,
        future_rate_residual_scale: float = 0.35,
    ):
        super().__init__()
        expected = 2 * num_trps * antennas_per_trp
        if pmi_real_dim != expected:
            raise ValueError(f"pmi_real_dim={pmi_real_dim} != {expected}")
        self.pmi_real_dim = pmi_real_dim
        self.num_trps = num_trps
        self.antennas_per_trp = antennas_per_trp
        self.complex_dim = num_trps * antennas_per_trp
        self.eta = float(eta)
        self.fairness_alpha = float(fairness_alpha)
        self.eps = float(eps)
        self.learned_correction_scale = float(learned_correction_scale)
        self.future_rate_residual_scale = float(future_rate_residual_scale)

        self.pooled_hw = (3, 2)
        cnn_flat_dim = 48 * self.pooled_hw[0] * self.pooled_hw[1]
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(self.pooled_hw),
        )
        self.scalar_proj = nn.Sequential(nn.Linear(7, 48), nn.LayerNorm(48), nn.GELU())
        self.corr_proj = nn.Sequential(nn.Linear(6, 32), nn.LayerNorm(32), nn.GELU())
        self.embed = nn.Sequential(
            nn.Linear(cnn_flat_dim + 48 + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ssa = nn.Sequential(
            SequenceShuffleAttention(embed_dim, groups=8, num_heads=4, dropout=dropout),
            SequenceShuffleAttention(embed_dim, groups=8, num_heads=4, dropout=dropout),
        )
        self.future_rate_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 7),
            nn.Linear(embed_dim + 7, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        self.eta_gate_head = nn.Sequential(
            nn.LayerNorm(embed_dim + 7),
            nn.Linear(embed_dim + 7, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )
        self.selection_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2 + 7 + 6),
            nn.Linear(embed_dim * 2 + 7 + 6, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        nn.init.zeros_(self.future_rate_head[-1].weight)
        nn.init.zeros_(self.future_rate_head[-1].bias)
        nn.init.zeros_(self.eta_gate_head[-1].weight)
        nn.init.zeros_(self.eta_gate_head[-1].bias)
        nn.init.zeros_(self.selection_head[-1].weight)
        nn.init.zeros_(self.selection_head[-1].bias)

    def _pmi_to_grid(self, now_pmi: torch.Tensor, future_pmi: torch.Tensor) -> torch.Tensor:
        b, k, _ = now_pmi.shape
        c = self.complex_dim
        nr = now_pmi[:, :, :c].reshape(b * k, 1, self.num_trps, self.antennas_per_trp)
        ni = now_pmi[:, :, c:].reshape(b * k, 1, self.num_trps, self.antennas_per_trp)
        fr = future_pmi[:, :, :c].reshape(b * k, 1, self.num_trps, self.antennas_per_trp)
        fi = future_pmi[:, :, c:].reshape(b * k, 1, self.num_trps, self.antennas_per_trp)
        return torch.cat([nr, ni, fr, fi], dim=1)

    def _corr_features(self, now_pmi: torch.Tensor, future_pmi: torch.Tensor) -> torch.Tensor:
        b, k, _ = now_pmi.shape
        corr_now = torch_complex_correlation_matrix(now_pmi, self.eps)
        corr_fut = torch_complex_correlation_matrix(future_pmi, self.eps)
        eye = torch.eye(k, dtype=torch.bool, device=now_pmi.device).unsqueeze(0)
        corr_now = corr_now.masked_fill(eye, 0.0)
        corr_fut = corr_fut.masked_fill(eye, 0.0)
        denom = float(max(k - 1, 1))
        top_count = max(min(k // 10, 8), 1)
        now_top = torch.topk(corr_now, k=top_count, dim=-1).values.mean(dim=-1, keepdim=True)
        fut_top = torch.topk(corr_fut, k=top_count, dim=-1).values.mean(dim=-1, keepdim=True)
        return torch.cat(
            [
                corr_now.sum(dim=-1, keepdim=True) / denom,
                corr_now.max(dim=-1, keepdim=True).values,
                now_top,
                corr_fut.sum(dim=-1, keepdim=True) / denom,
                corr_fut.max(dim=-1, keepdim=True).values,
                fut_top,
            ],
            dim=-1,
        )

    def _split(self, x: torch.Tensor):
        d = self.pmi_real_dim
        now_pmi = x[:, :, :d]
        now_rate = x[:, :, d : d + 1]
        future_pmi = x[:, :, d + 1 : d + 1 + d]
        future_rate = x[:, :, d + 1 + d : d + 1 + d + 1]
        hist = x[:, :, d + 1 + d + 1 : d + 1 + d + 2]
        speed = x[:, :, d + 1 + d + 2 : d + 1 + d + 3]
        return now_pmi, now_rate, future_pmi, future_rate, hist, speed

    def forward_with_aux(
        self,
        x: torch.Tensor,
        eta_override: float | torch.Tensor | None = None,
        emergency_priority: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,K,F), got {tuple(x.shape)}")
        b, k, _ = x.shape
        now_pmi, now_rate, future_pmi, future_rate, hist, speed = self._split(x)
        hist_safe = torch.clamp(hist, min=self.eps)
        def norm_user(v: torch.Tensor) -> torch.Tensor:
            return (v - v.mean(dim=1, keepdim=True)) / (v.std(dim=1, keepdim=True, unbiased=False) + self.eps)

        eta_base = self.eta if eta_override is None else eta_override
        if isinstance(eta_base, torch.Tensor):
            eta_tensor = eta_base.to(device=x.device, dtype=x.dtype)
        else:
            eta_tensor = torch.tensor(float(eta_base), device=x.device, dtype=x.dtype)
        eta_tensor = torch.clamp(eta_tensor, 1e-4, 1.0 - 1e-4)
        base_q = (eta_tensor * now_rate + (1.0 - eta_tensor) * future_rate) / torch.pow(hist_safe, self.fairness_alpha)
        scalar_raw = torch.cat(
            [
                norm_user(now_rate),
                norm_user(future_rate),
                norm_user(hist_safe),
                speed,
                norm_user(future_rate - now_rate),
                norm_user(torch.rsqrt(hist_safe)),
                norm_user(base_q),
            ],
            dim=-1,
        )
        corr_raw = self._corr_features(now_pmi, future_pmi)
        grid = self._pmi_to_grid(now_pmi, future_pmi)
        cnn_feat = self.cnn(grid).flatten(1).reshape(b, k, -1)
        z = self.embed(torch.cat([cnn_feat, self.scalar_proj(scalar_raw), self.corr_proj(corr_raw)], dim=-1))
        z = self.ssa(z)

        rate_scale = future_rate.std(dim=1, keepdim=True, unbiased=False).clamp(min=0.05)
        delta_norm = self.future_rate_head(torch.cat([z, scalar_raw], dim=-1))
        calibrated_future = torch.clamp(
            future_rate + self.future_rate_residual_scale * rate_scale * torch.tanh(delta_norm),
            min=0.0,
        )
        base_logit = torch.log(eta_tensor / (1.0 - eta_tensor))
        eta_delta = self.eta_gate_head(torch.cat([z, scalar_raw], dim=-1))
        eta_user = torch.sigmoid(base_logit + eta_delta)
        q_cal = (eta_user * now_rate + (1.0 - eta_user) * calibrated_future) / torch.pow(hist_safe, self.fairness_alpha)
        if emergency_priority is not None:
            priority = emergency_priority.to(device=x.device, dtype=x.dtype)
            if priority.ndim == 2:
                priority = priority.unsqueeze(-1)
            if priority.shape != q_cal.shape:
                raise ValueError(
                    f"emergency_priority must have shape {tuple(q_cal.shape[:2])} or {tuple(q_cal.shape)}, "
                    f"got {tuple(priority.shape)}"
                )
            q_cal = q_cal * torch.clamp(priority, min=self.eps)
        q_prior = torch.log(torch.clamp(q_cal.squeeze(-1), min=self.eps))
        q_prior = norm_user(q_prior.unsqueeze(-1)).squeeze(-1)

        global_context = z.mean(dim=1, keepdim=True).expand_as(z)
        correction = self.selection_head(torch.cat([z, global_context, scalar_raw, corr_raw], dim=-1)).squeeze(-1)
        correction = correction - correction.mean(dim=1, keepdim=True)
        correction = correction / correction.std(dim=1, keepdim=True, unbiased=False).clamp(min=0.5)
        logits = q_prior + self.learned_correction_scale * correction
        return {
            "logits": logits,
            "future_rate_hat": calibrated_future.squeeze(-1),
            "q_prior": q_prior,
            "learned_correction": correction,
            "eta_user": eta_user.squeeze(-1),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(x)["logits"]
