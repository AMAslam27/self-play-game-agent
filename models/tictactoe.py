"""Tic-Tac-Toe Q-network built from reusable hidden blocks."""

from collections.abc import Sequence

from torch import Tensor, nn

from models.blocks import build_blocks


class TicTacToeQNetwork(nn.Module):
    """Map nine board features to nine unrestricted action-value estimates.

    Inputs are floating-point tensors shaped (9,) or (batch_size, 9), on the
    same device as the model. Use the adapter's agent-relative board encoding.
    Action selection and learning targets must mask illegal actions separately.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (64, 64),
        block_types: Sequence[str] = ("dense", "dense"),
        activations: Sequence[str] = ("relu", "relu"),
        dropout_probability: float = 0.1,
    ):
        super().__init__()
        self.blocks = build_blocks(
            input_size=9,
            hidden_sizes=hidden_sizes,
            block_types=block_types,
            activations=activations,
            dropout_probability=dropout_probability,
        )
        # No activation or dropout here: Q-values may be positive or negative.
        self.output = nn.Linear(hidden_sizes[-1], 9)

    def forward(self, board: Tensor) -> Tensor:
        return self.output(self.blocks(board))
