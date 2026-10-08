"""Entry point for running TicTacToe games.

Examples:
    python runner.py                          # human vs human
    python runner.py --x human --o random     # you (X) vs random bot
    python runner.py --x random --o random --games 1000 --quiet
"""

import argparse
import logging
import random
from contextlib import closing
from pathlib import Path

from evaluation.artifacts import display_project_path, save_run_plot
from evaluation.database import DEFAULT_DB_PATH, connect_database
from evaluation.model_registry import (
    DEFAULT_REGISTRY_PATH,
    checkpoint_hash,
    connect_registry,
)
from evaluation.recorder import ModelIdentity, RunRecorder
from games.tictactoe.play import human_policy, play_game, random_policy
from games.tictactoe.rules import EMPTY, PLAYER_O, PLAYER_X
from players.checkpoint_selection import resolve_checkpoint
from players.checkpoints import load_model_checkpoint
from players.minimax import minimax_policy
from players.saved_agent import SavedAgentPlayer
from players.tictactoe import TicTacToePlayerAdapter, build_tictactoe_model

logger = logging.getLogger(__name__)

# Fixed players; saved DQN players are constructed separately for each seat.
POLICIES = {"human": human_policy, "random": random_policy, "minimax": minimax_policy}


def parse_args():
    parser = argparse.ArgumentParser(description="Run TicTacToe games.")
    parser.add_argument(
        "--x",
        choices=(*POLICIES, "dqn"),
        default="human",
        help="player X (moves first)",
    )
    parser.add_argument(
        "--o", choices=(*POLICIES, "dqn"), default="human", help="player O"
    )
    for seat in ("x", "o"):
        source = parser.add_mutually_exclusive_group()
        source.add_argument(
            f"--{seat}-agent",
            choices=("latest", "best"),
            help=f"Select a registered DQN for {seat.upper()}",
        )
        source.add_argument(
            f"--{seat}-checkpoint",
            type=Path,
            help=f"Explicit DQN checkpoint for {seat.upper()}",
        )
        parser.add_argument(
            f"--{seat}-protocol", help="Evaluation protocol ID when selecting best"
        )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Model registry database",
    )
    parser.add_argument(
        "--agent-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--games", type=int, default=1, help="number of games to play")
    parser.add_argument(
        "--quiet", action="store_true", help="don't print boards or per-game results"
    )
    parser.add_argument(
        "--plot-file",
        help="override the automatic timestamped chart path",
    )
    parser.add_argument(
        "--db-file",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite results database (default: project results/games.sqlite3)",
    )
    parser.add_argument("--seed", type=int, help="random seed for each game batch")
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be a positive integer")
    for seat in ("x", "o"):
        agent, checkpoint, protocol = (
            getattr(args, f"{seat}_{field}")
            for field in ("agent", "checkpoint", "protocol")
        )
        if getattr(args, seat) == "dqn" and agent is None and checkpoint is None:
            parser.error(f"--{seat} dqn requires --{seat}-agent or --{seat}-checkpoint")
        if getattr(args, seat) != "dqn" and any(
            value is not None for value in (agent, checkpoint, protocol)
        ):
            parser.error(f"Saved-agent options for {seat.upper()} require --{seat} dqn")
        if protocol is not None and agent != "best":
            parser.error(f"--{seat}-protocol requires --{seat}-agent best")
    return args


def prepare_player(args, seat):
    """Load a saved policy once, or return a fixed player and no model identity."""
    name = getattr(args, seat)
    if name != "dqn":
        return POLICIES[name], None
    selection = getattr(args, f"{seat}_checkpoint") or getattr(args, f"{seat}_agent")
    protocol_id = getattr(args, f"{seat}_protocol")
    expected_hash = None
    if isinstance(selection, str) and selection in ("latest", "best"):
        with closing(connect_registry(args.registry)) as registry:
            path = resolve_checkpoint(
                selection, connection=registry, protocol_id=protocol_id
            )
            record = registry.execute(
                "SELECT checkpoint_hash, protocol_id FROM model_checkpoints WHERE path=? AND is_final=1",
                (str(path),),
            ).fetchone()
            if record is None:
                raise ValueError("Selected checkpoint is no longer registered")
            expected_hash = record["checkpoint_hash"]
            protocol_id = record["protocol_id"]
    else:
        path = resolve_checkpoint(selection)
    identity = checkpoint_hash(path)
    if expected_hash is not None and identity != expected_hash:
        raise ValueError("Selected checkpoint changed after resolution")
    loaded = load_model_checkpoint(path, build_tictactoe_model, args.agent_device)
    if checkpoint_hash(path) != identity:
        raise ValueError("Checkpoint changed while loading the model")
    experiment_name = str(
        loaded.config.get("experiment", {}).get("name", loaded.run_id)
    )
    model = ModelIdentity(
        loaded.run_id, experiment_name, str(path), identity, str(selection), protocol_id
    )
    print(
        f"{seat.upper()} agent: {experiment_name} | run {loaded.run_id} | "
        f"checkpoint {display_project_path(path)} | SHA-256 {identity}"
    )
    return SavedAgentPlayer(
        loaded.model, TicTacToePlayerAdapter(), device=args.agent_device
    ), model


def print_progress(completed, total):
    if completed % 100 == 0 or completed == total:
        print(f"{completed}/{total} games completed", flush=True)


def print_summary(results):
    total = sum(results.values())
    print(f"\nResults over {total} game(s):")
    print(f"  X wins: {results[PLAYER_X]}")
    print(f"  O wins: {results[PLAYER_O]}")
    print(f"  Draws:  {results[EMPTY]}")


def finish_after_error(recorder, status):
    """Try to preserve the final status without hiding the original error."""
    try:
        recorder.finish_run(status)
    except Exception:
        logger.exception("Could not save status for run %s", recorder.run_id)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        policy_x, model_x = prepare_player(args, "x")
        policy_o, model_o = prepare_player(args, "o")
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        raise SystemExit(f"Cannot prepare saved agent: {error}") from error
    interactive = "human" in (args.x, args.o)
    results = {PLAYER_X: 0, PLAYER_O: 0, EMPTY: 0}

    if not args.quiet:
        print(f"X: {args.x} | O: {args.o}. Squares are numbered 1-9.")

    with closing(connect_database(args.db_file)) as connection:
        batch_number = 0
        while True:
            batch_number += 1
            batch_results = {PLAYER_X: 0, PLAYER_O: 0, EMPTY: 0}
            recorder = RunRecorder(connection)
            if args.seed is not None:
                random.seed(args.seed)
            run_id = recorder.start_run(
                args.x,
                args.o,
                args.games,
                seed=args.seed,
                model_x=model_x,
                model_o=model_o,
            )
            logger.info(
                "Started run %s: X=%s, O=%s, games=%s",
                run_id,
                args.x,
                args.o,
                args.games,
            )

            try:
                quit_early = False
                for game_number in range(1, args.games + 1):
                    winner = play_game(policy_x, policy_o, verbose=not args.quiet)
                    if winner is None:
                        quit_early = True
                        break

                    recorder.record_game(game_number, winner)
                    results[winner] += 1
                    batch_results[winner] += 1
                    if args.games > 1:
                        print_progress(game_number, args.games)

                status = "abandoned" if quit_early else "completed"
                recorder.finish_run(status)
                logger.info("Finished run %s: %s", run_id, status)
            except KeyboardInterrupt:
                finish_after_error(recorder, "interrupted")
                logger.warning("Run %s interrupted", run_id)
                raise
            except Exception:
                finish_after_error(recorder, "failed")
                logger.exception("Run %s failed", run_id)
                raise

            output_path = args.plot_file
            if output_path is not None and batch_number > 1:
                path = Path(output_path)
                output_path = path.with_name(f"{path.stem}_run-{run_id}{path.suffix}")
            saved_path = save_run_plot(
                batch_results, args.x, args.o, recorder, output_path=output_path
            )
            if saved_path is not None:
                print(f"Chart saved to: {display_project_path(saved_path)}")

            if quit_early or not interactive:
                break
            again = input("Play again? (y/n): ").strip().lower()
            if again not in ("y", "yes"):
                break

    if args.games > 1 or args.quiet or sum(results.values()) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
