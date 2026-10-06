"""Reusable fully connected blocks and a builder for configurable networks."""

from collections.abc import Callable, Sequence

from torch import Tensor, nn


def _validate_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Layer sizes must be positive integers")


def _activation(name: str) -> nn.Module:
    """Create a fresh activation module for each block."""
    activations: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "identity": nn.Identity,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name!r}")
    return activations[name]()


class DenseBlock(nn.Sequential):
    """Linear transformation followed by an activation."""

    def __init__(self, input_size: int, output_size: int, activation: str):
        _validate_size(input_size)
        _validate_size(output_size)
        super().__init__(nn.Linear(input_size, output_size), _activation(activation))


class DropoutBlock(nn.Sequential):
    """Dense block followed by dropout; dropout is disabled in eval mode."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str,
        dropout_probability: float,
    ):
        if not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Dropout probability must be in [0, 1)")
        super().__init__(
            DenseBlock(input_size, output_size, activation),
            nn.Dropout(dropout_probability),
        )


class ResidualBlock(nn.Module):
    """Return x + Linear(activation(Linear(x))), without a projection.

    The input and output widths must match for the skip addition. The
    activation is inside the learned branch; there is no activation after
    addition, preserving the identity path when that branch outputs zero.
    """

    def __init__(self, input_size: int, output_size: int, activation: str):
        super().__init__()
        _validate_size(input_size)
        _validate_size(output_size)
        if input_size != output_size:
            raise ValueError("Residual blocks require matching input and output sizes")
        self.branch = nn.Sequential(
            nn.Linear(input_size, output_size),
            _activation(activation),
            nn.Linear(output_size, output_size),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.branch(inputs)


BlockFactory = Callable[[int, int, str, float], nn.Module]

BLOCK_FACTORIES: dict[str, BlockFactory] = {
    "dense": lambda input_size, output_size, activation, _dropout: DenseBlock(
        input_size, output_size, activation
    ),
    "dropout": DropoutBlock,
    "residual": lambda input_size, output_size, activation, _dropout: ResidualBlock(
        input_size, output_size, activation
    ),
}


def build_blocks(
    input_size: int,
    hidden_sizes: Sequence[int],
    block_types: Sequence[str],
    activations: Sequence[str],
    dropout_probability: float = 0.1,
) -> nn.Sequential:
    """Build hidden blocks from matching lists, in order.

    Each hidden size is that block's output width. Its input width comes from
    the preceding block (or input_size for the first block). Supported block
    types are dense, dropout, and residual. Supported activations are relu,
    tanh, gelu, and identity. The shared dropout probability applies only to
    dropout blocks. The caller adds a task-specific output head separately.
    """
    _validate_size(input_size)
    if not hidden_sizes:
        raise ValueError("At least one hidden block is required")
    if not len(hidden_sizes) == len(block_types) == len(activations):
        raise ValueError(
            "hidden_sizes, block_types, and activations must have equal lengths"
        )
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("Dropout probability must be in [0, 1)")

    blocks: list[nn.Module] = []
    previous_size = input_size

    for output_size, block_type, activation in zip(
        hidden_sizes, block_types, activations, strict=True
    ):
        factory = BLOCK_FACTORIES.get(block_type)
        if factory is None:
            raise ValueError(f"Unsupported block type: {block_type!r}")

        block = factory(
            previous_size,
            output_size,
            activation,
            dropout_probability,
        )
        blocks.append(block)
        previous_size = output_size

    return nn.Sequential(*blocks)
