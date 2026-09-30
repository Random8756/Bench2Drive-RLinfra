"""Two-hot distributional value utilities with optional symlog scaling."""

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: sign(x) * ln(|x| + 1).

    Compresses large magnitudes while preserving sign.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TwoHotDistributional:
    """Two-hot distributional value representation.

    Discretises a continuous value range [v_min, v_max] into *num_bins*
    evenly-spaced bins (optionally in symlog space).  Provides:

    * ``encode(values)`` – scalar → two-hot target vector
    * ``decode(logits)`` – network logits → scalar expected value
    * ``loss(logits, target_values, weights)`` – cross-entropy loss

    Args:
        num_bins:   Number of bins (K).
        v_min:      Minimum value in **original** (pre-symlog) space.
        v_max:      Maximum value in **original** (pre-symlog) space.
        use_symlog: If True, bin edges are placed uniformly in symlog space.
    """

    def __init__(
        self,
        num_bins: int = 255,
        v_min: float = -300.0,
        v_max: float = 800.0,
        use_symlog: bool = True,
    ):
        self.num_bins = num_bins
        self.use_symlog = use_symlog
        self.v_min_raw = v_min
        self.v_max_raw = v_max

        # Transformed limits
        if use_symlog:
            self.v_min_t = symlog(torch.tensor(v_min)).item()
            self.v_max_t = symlog(torch.tensor(v_max)).item()
        else:
            self.v_min_t = v_min
            self.v_max_t = v_max

        # Bin centres – evenly spaced in transformed space
        self.bin_centers: torch.Tensor = torch.linspace(
            self.v_min_t, self.v_max_t, num_bins
        )
        self.bin_width: float = (self.v_max_t - self.v_min_t) / (num_bins - 1)

    def to(self, device: torch.device) -> "TwoHotDistributional":
        """Move internal tensors to *device* (in-place, returns self)."""
        self.bin_centers = self.bin_centers.to(device)
        return self

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Encode scalar values into two-hot probability vectors.

        Args:
            values: shape ``(batch, 1)`` or ``(batch,)``.

        Returns:
            Two-hot vectors, shape ``(batch, num_bins)``.
        """
        values = values.float().squeeze(-1)  # (batch,)

        # Transform
        if self.use_symlog:
            values = symlog(values)

        # Clamp to valid range
        values = values.clamp(self.v_min_t, self.v_max_t)

        # Continuous bin index
        norm = (values - self.v_min_t) / self.bin_width
        lower = norm.floor().long().clamp(0, self.num_bins - 2)
        upper = lower + 1

        # Interpolation weights
        upper_w = (norm - lower.float()).unsqueeze(1)   # (batch, 1)
        lower_w = 1.0 - upper_w

        # Build two-hot
        twohot = torch.zeros(
            values.shape[0], self.num_bins,
            device=values.device, dtype=values.dtype,
        )
        twohot.scatter_(1, lower.unsqueeze(1), lower_w)
        twohot.scatter_(1, upper.unsqueeze(1), upper_w)
        return twohot

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode network logits to scalar expected values.

        Args:
            logits: shape ``(batch, num_bins)``.

        Returns:
            Scalar values, shape ``(batch, 1)``.
        """
        probs = F.softmax(logits, dim=-1)
        # Expected value in transformed space
        expected = (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1, keepdim=True)

        if self.use_symlog:
            expected = symexp(expected)
        return expected

    def decode_for_actor(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode logits to expected value **in transformed (symlog) space**.

        Unlike :meth:`decode`, this method does **not** apply ``symexp``,
        so the gradient path through the actor remains
        ``softmax → weighted-sum`` only — no exponential amplification.

        Since ``symlog`` is monotonically increasing, maximising the
        expected symlog-Q is equivalent (direction-wise) to maximising Q
        in the original space, but with much more stable gradients.

        Use this method exclusively for computing the actor (policy
        gradient) loss.  For TD targets, logging, PER priorities, etc.,
        keep using :meth:`decode`.

        Args:
            logits: shape ``(batch, num_bins)``.

        Returns:
            Expected value in transformed space, shape ``(batch, 1)``.
        """
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1, keepdim=True)

    def loss(
        self,
        logits: torch.Tensor,
        target_values: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        """Cross-entropy loss against two-hot encoded targets.

        Args:
            logits:        shape ``(batch, num_bins)``  – raw network output.
            target_values: shape ``(batch, 1)``         – scalar TD targets.
            weights:       shape ``(batch, 1)`` or None – optional PER IS weights.

        Returns:
            Scalar loss (mean-reduced).
        """
        targets = self.encode(target_values)  # (batch, num_bins)
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample = -(targets * log_probs).sum(dim=-1, keepdim=True)  # (batch, 1)

        if weights is not None:
            return (weights * per_sample).mean()
        return per_sample.mean()
