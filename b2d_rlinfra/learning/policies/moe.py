"""Top-k gated mixture-of-experts MLP components."""

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    """MLP expert whose output projection has no trailing activation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(activation_fn())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEMLP(nn.Module):
    """Top-k sparsely-gated Mixture-of-Experts MLP.

    Drop-in replacement for ``create_mlp(input_dim, output_dim, hidden, ...)``.

    Output shape: ``(batch, output_dim)``.

    After every forward pass the following are populated for logging:

    - ``self.last_aux_loss``: scalar tensor, Switch-style load-balancing loss
      ``num_experts * mean(importance * load)``.  Add this to the total loss
      scaled by ``aux_loss_weight``.
    - ``self.last_gate_stats``: plain-Python dict with:
        * ``usage``: list of length ``num_experts``, fraction of tokens in the
          current batch routed to each expert through top-k.
        * ``gate_entropy``: scalar float, mean Shannon entropy of the softmax
          gate distribution (before top-k).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: List[int],
        num_experts: int = 4,
        top_k: int = 2,
        activation_fn: Type[nn.Module] = nn.ReLU,
        noisy_gating: bool = True,
        output_activation: Optional[Type[nn.Module]] = None,
    ):
        super().__init__()
        assert num_experts >= 1, "num_experts must be >= 1"
        assert 1 <= top_k <= num_experts, "top_k must satisfy 1 <= top_k <= num_experts"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating

        self.experts = nn.ModuleList(
            [
                _Expert(input_dim, output_dim, hidden, activation_fn)
                for _ in range(num_experts)
            ]
        )

        self.w_gate = nn.Linear(input_dim, num_experts, bias=False)
        if noisy_gating:
            self.w_noise = nn.Linear(input_dim, num_experts, bias=False)
        else:
            self.w_noise = None

        self.output_activation = output_activation() if output_activation is not None else None

        # Buffers populated on forward(); initialized to zeros so they are
        # safe to read before the first forward call.
        self.register_buffer("_aux_loss_buf", torch.zeros((), dtype=torch.float32), persistent=False)
        self.last_gate_stats = {
            "usage": [0.0] * num_experts,
            "gate_entropy": 0.0,
        }

    @property
    def last_aux_loss(self) -> torch.Tensor:
        """Scalar tensor (may require grad) holding the most recent aux loss."""
        return self._aux_loss_buf

    def _compute_gate(self, x: torch.Tensor):
        logits = self.w_gate(x)
        if self.training and self.noisy_gating and self.w_noise is not None:
            raw_noise = self.w_noise(x)
            noise_std = F.softplus(raw_noise) + 1e-4
            logits = logits + noise_std * torch.randn_like(logits)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)

        gate_logits = self._compute_gate(x)
        # Full softmax over all experts (used for aux loss + gate entropy).
        probs = F.softmax(gate_logits, dim=-1)

        # Top-k routing.
        topk_val, topk_idx = gate_logits.topk(self.top_k, dim=-1)
        topk_gate = F.softmax(topk_val, dim=-1)

        out = x.new_zeros((batch, self.output_dim))

        # Bucket tokens by expert to avoid running every expert on the full
        # batch.  top_k > 1 is handled by iterating over the k-th slot.
        for k in range(self.top_k):
            idx_k = topk_idx[:, k]          # (B,)
            gate_k = topk_gate[:, k].unsqueeze(-1)  # (B, 1)
            for e_id in range(self.num_experts):
                mask = idx_k == e_id
                if mask.any():
                    expert_in = x[mask]
                    expert_out = self.experts[e_id](expert_in)
                    out[mask] = out[mask] + gate_k[mask] * expert_out

        # Switch-style load-balancing loss.
        # importance_e = mean over batch of probability mass assigned to expert e.
        importance = probs.mean(dim=0)  # (num_experts,)
        # load_e = fraction of tokens that selected expert e in top-k.
        with_grad_load = torch.zeros_like(importance)
        for k in range(self.top_k):
            # One-hot over experts, averaged over batch.
            onehot = F.one_hot(topk_idx[:, k], num_classes=self.num_experts).float()
            with_grad_load = with_grad_load + onehot.mean(dim=0)
        load = with_grad_load / self.top_k  # fraction, shape (num_experts,)
        aux_loss = self.num_experts * (importance * load).sum()
        self._aux_loss_buf = aux_loss

        # Detached gate statistics for logging.
        with torch.no_grad():
            gate_entropy = -(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1).mean()
            self.last_gate_stats = {
                "usage": load.detach().cpu().tolist(),
                "gate_entropy": float(gate_entropy.item()),
            }

        if self.output_activation is not None:
            out = self.output_activation(out)
        return out
