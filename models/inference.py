"""Shared device resolution and legal greedy selection for PyTorch models."""

from collections.abc import Sequence

import torch
from torch import Tensor


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve auto to CUDA when available, otherwise CPU."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def select_greedy_q_action(output: Tensor, legal_actions: Sequence[int]) -> int:
    """Choose the largest legal Q-value from a (1, actions) output tensor.

    Ties choose the lowest output index. Only legal values must be finite;
    illegal outputs are ignored rather than allowed to influence selection.
    """
    if output.ndim != 2 or output.shape[0] != 1 or not output.is_floating_point():
        raise ValueError(
            "Q-network output must be a floating tensor shaped (1, actions)"
        )
    if not legal_actions:
        raise ValueError("Cannot select an action without legal moves")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < output.shape[1]
        for index in legal_actions
    ):
        raise ValueError("Legal actions must be valid Q-network output indices")
    indices = torch.tensor(sorted(set(legal_actions)), device=output.device)
    values = output[0, indices]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Legal Q-values must be finite")
    return int(indices[values.argmax()].item())
