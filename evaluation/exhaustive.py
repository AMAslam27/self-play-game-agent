"""Explore every opponent response to a fixed greedy Tic-Tac-Toe policy."""

import argparse
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from evaluation.model_registry import checkpoint_hash
from evaluation.tictactoe_matches import Policy
from games.tictactoe.rules import PLAYER_O, PLAYER_X, TicTacToe
from models.inference import resolve_device
from players.load_checkpoints import load_model_checkpoint
from players.saved_agent import SavedAgentPlayer
from players.tictactoe_player import TicTacToePlayerAdapter, build_tictactoe_model
from training.metrics import write_json
from utils.timestamps import utc_now


def runtime_metadata(device: str) -> dict[str, str]:
    resolved = resolve_device(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(resolved),
        "gpu": torch.cuda.get_device_name(resolved) if resolved.type == "cuda" else "",
    }


def load_player(path: Path, device: str) -> tuple[Policy, dict[str, Any]]:
    """Load inference weights and verify their identity across loading."""
    before = checkpoint_hash(path)
    loaded = load_model_checkpoint(path, build_tictactoe_model, device)
    if checkpoint_hash(path) != before:
        raise ValueError(f"Checkpoint changed while loading: {path}")
    return SavedAgentPlayer(loaded.model, TicTacToePlayerAdapter(), device=device), {
        "run_id": loaded.run_id,
        "checkpoint_path": str(loaded.checkpoint_path),
        "checkpoint_hash": before,
        "training_method": loaded.config["environment"]["opponent"],
        "training_seed": loaded.config["experiment"]["seed"],
    }


@dataclass(frozen=True)
class ExhaustiveResult:
    seat: str
    wins: int
    draws: int
    losses: int
    states: int
    losing_sequence: tuple[int, ...] | None

    @property
    def unbeatable(self) -> bool:
        return self.losses == 0


def assess_policy(policy: Policy, seat: str) -> ExhaustiveResult:
    """Enumerate terminal move sequences, stopping immediately at game end.

    The policy must be deterministic, legal, and depend only on the board/turn.
    Opponent branches are not probability weighted. Squares use indices 0–8.
    """
    if seat not in ("x", "o"):
        raise ValueError("Seat must be x or o")
    player = PLAYER_X if seat == "x" else PLAYER_O
    outcomes = {1: 0, 0: 0, -1: 0}
    states: set[tuple[tuple[int, ...], int]] = set()
    actions: dict[tuple[tuple[int, ...], int], int] = {}
    losing_sequence = None

    def visit(game: TicTacToe, moves: tuple[int, ...]) -> None:
        nonlocal losing_sequence
        state = (tuple(game.board), game.current_player)
        states.add(state)
        if game.is_terminal():
            outcome = game.winner() * player
            outcomes[outcome] += 1
            if outcome == -1 and losing_sequence is None:
                losing_sequence = moves
            return
        legal = game.legal_actions()
        if game.current_player == player:
            if state not in actions:
                # Give policies a disposable copy so they cannot mutate traversal.
                view = TicTacToe()
                view.board = list(game.board)
                view.current_player = game.current_player
                action = policy(view)
                if (
                    isinstance(action, bool)
                    or not isinstance(action, int)
                    or action not in legal
                ):
                    raise ValueError("Policy must return a legal integer action")
                actions[state] = action
            branches = [actions[state]]
        else:
            branches = legal
        for action in branches:
            child = TicTacToe()
            child.board = list(game.board)
            child.current_player = game.current_player
            child.step(action)
            visit(child, (*moves, action))

    visit(TicTacToe(), ())
    return ExhaustiveResult(
        seat, outcomes[1], outcomes[0], outcomes[-1], len(states), losing_sequence
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evaluator": "tictactoe-exhaustive-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": args.device,
        "runtime": runtime_metadata(args.device),
        "scope": "Empty board; fixed greedy policy; all legal opponent responses",
        "counts": "Terminal move sequences, not sampled probabilities",
        "square_indices": "0-8, row major",
        "models": [],
    }
    path = args.output / "exhaustive.json"
    write_json(path, manifest)
    try:
        seen = set()
        for checkpoint in args.checkpoints:
            policy, identity = load_player(checkpoint, args.device)
            if identity["checkpoint_hash"] in seen:
                raise ValueError("Duplicate checkpoint")
            seen.add(identity["checkpoint_hash"])
            seats = []
            for seat in ("x", "o"):
                result = assess_policy(policy, seat)
                seats.append({**asdict(result), "unbeatable": result.unbeatable})
            manifest["models"].append({**identity, "seats": seats})
            write_json(path, manifest)
            print(
                f"{identity['run_id']}: "
                + ", ".join(
                    f"{row['seat'].upper()} unbeatable={row['unbeatable']}"
                    for row in seats
                ),
                flush=True,
            )
        manifest["status"] = "completed"
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_at"] = utc_now()
        write_json(path, manifest)


if __name__ == "__main__":
    main()
