"""Scalar transforms and two-hot categorical value representations."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def symlog(values: torch.Tensor) -> torch.Tensor:
    """Symmetric log transform."""
    return torch.sign(values) * torch.log1p(torch.abs(values))


def symexp(values: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(values) * torch.expm1(torch.abs(values))


def apply_value_transform(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply the configured value transform in transformed/support space."""
    if transform == "identity":
        return values
    if transform == "symlog":
        return symlog(values)
    raise ValueError(f"Unsupported value transform: {transform}")


def inverse_value_transform(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Invert a transformed-space value back to raw scalar space."""
    if transform == "identity":
        return values
    if transform == "symlog":
        return symexp(values)
    raise ValueError(f"Unsupported value transform: {transform}")


def build_value_support(
    num_bins: int,
    support_min: float,
    support_max: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a linearly spaced support in transformed space."""
    if num_bins < 2:
        raise ValueError(f"value_num_bins must be >= 2, got {num_bins}")
    if support_max <= support_min:
        raise ValueError(
            f"value_support_max must be greater than value_support_min, got "
            f"{support_min} >= {support_max}"
        )
    return torch.linspace(
        float(support_min),
        float(support_max),
        steps=int(num_bins),
        device=device,
        dtype=dtype,
    )


def _support_view(support: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Reshape a 1D support for broadcasting against logits tensors."""
    return support.view(*([1] * (target_ndim - 1)), -1)


def logits_to_transformed_scalar(
    logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Decode logits to the support-space expected value."""
    probs = torch.softmax(logits, dim=-1)
    return torch.sum(probs * _support_view(support, logits.ndim), dim=-1)


def logits_to_scalar(
    logits: torch.Tensor,
    support: torch.Tensor,
    transform: str,
) -> torch.Tensor:
    """Decode categorical value logits into raw scalar value space."""
    transformed = logits_to_transformed_scalar(logits, support)
    return inverse_value_transform(transformed, transform)


def scalar_to_twohot(
    transformed_values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """
    Project transformed scalar values onto neighboring support bins.

    Boundary handling is fixed:
    - values below support_min => all mass on the first bin
    - values above support_max => all mass on the last bin
    - values inside support => linear interpolation between adjacent bins
    """
    if support.ndim != 1:
        raise ValueError(f"support must be 1D, got shape {tuple(support.shape)}")
    if support.numel() < 2:
        raise ValueError("support must contain at least 2 bins")

    flat_values = transformed_values.reshape(-1).to(dtype=support.dtype)
    clipped = flat_values.clamp(min=support[0], max=support[-1])

    right_idx = torch.searchsorted(support, clipped, right=False)
    right_idx = torch.clamp(right_idx, 1, support.numel() - 1)
    left_idx = right_idx - 1

    left_support = support[left_idx]
    right_support = support[right_idx]
    denom = torch.clamp(
        right_support - left_support,
        min=torch.finfo(clipped.dtype).eps,
    )

    right_weight = (clipped - left_support) / denom
    left_weight = 1.0 - right_weight

    target = torch.zeros(
        flat_values.shape[0],
        support.numel(),
        device=clipped.device,
        dtype=clipped.dtype,
    )
    target.scatter_add_(1, left_idx.unsqueeze(-1), left_weight.unsqueeze(-1))
    target.scatter_add_(1, right_idx.unsqueeze(-1), right_weight.unsqueeze(-1))

    return target.view(*transformed_values.shape, support.numel())


def categorical_value_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """Soft-target cross-entropy for categorical value learning."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def categorical_value_stats(
    logits: torch.Tensor,
    support: torch.Tensor,
    transform: str,
    target_raw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute logging stats for categorical value learning.

    Returns:
        value_pred_mean_raw
        value_pred_mean_transformed
        value_target_mean_raw
        value_target_mean_transformed
    """
    pred_transformed = logits_to_transformed_scalar(logits, support)
    pred_raw = inverse_value_transform(pred_transformed, transform)
    target_transformed = apply_value_transform(target_raw, transform).clamp(
        min=support[0],
        max=support[-1],
    )
    return (
        pred_raw.mean(),
        pred_transformed.mean(),
        target_raw.mean(),
        target_transformed.mean(),
    )


def categorical_distribution_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean entropy of the categorical value distribution."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()
