"""Tic-Tac-Toe encoding and construction for saved inference players."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from games.tictactoe.rules import TicTacToe
from models.tictactoe import TicTacToeQNetwork


class TicTacToePlayerAdapter:
    """Encode each move from the current player's perspective, for either seat."""

    def encode(self, game: TicTacToe) -> Tensor:
        return torch.tensor(
            [[cell * game.current_player for cell in game.board]], dtype=torch.float32
        )

    def legal_actions(self, game: TicTacToe) -> Sequence[int]:
        return [] if game.is_terminal() else game.legal_actions()

    def decode_action(self, game: TicTacToe, index: int) -> int:
        if index not in self.legal_actions(game):
            raise ValueError("Cannot decode an illegal Tic-Tac-Toe action")
        return index


def build_tictactoe_model(config: Mapping[str, Any]) -> TicTacToeQNetwork:
    """Construct the saved architecture for the existing Tic-Tac-Toe format."""
    network = config.get("network")
    if not isinstance(network, Mapping):
        raise ValueError("Tic-Tac-Toe checkpoint must include network configuration")
    return TicTacToeQNetwork(**network)
