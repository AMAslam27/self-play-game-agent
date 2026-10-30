"""Compare checkpoints with explicitly frozen historical reference models."""

import argparse
from pathlib import Path
from typing import Any

from evaluation.exhaustive import load_player, runtime_metadata
from evaluation.tictactoe_matches import Policy
from games.tictactoe.play import play_game
from training.metrics import write_json
from utils.timestamps import utc_now


def compare_policies(policy: Policy, reference: Policy) -> list[dict[str, Any]]:
    """One deterministic game per seat; results are from the candidate's view."""
    rows = []
    for seat in ("x", "o"):
        moves: list[int | None] = []

        def recorded_policy(game, *, seat=seat, moves=moves):
            acting = (
                policy if (game.current_player == 1) == (seat == "x") else reference
            )
            action = acting(game)
            moves.append(action)
            return action

        winner = play_game(
            recorded_policy,
            recorded_policy,
            verbose=False,
        )
        if winner is None:
            raise ValueError("Evaluation policies cannot quit")
        outcome = winner if seat == "x" else -winner
        rows.append(
            {
                "seat": seat,
                "games": 1,
                "wins": int(outcome == 1),
                "draws": int(outcome == 0),
                "losses": int(outcome == -1),
                "moves": moves,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evaluator": "tictactoe-historical-v1",
        "status": "running",
        "started_at": utc_now(),
        "device": args.device,
        "runtime": runtime_metadata(args.device),
        "protocol": "Greedy lowest-index ties; one deterministic game per seat/reference",
        "references": [],
        "models": [],
    }
    path = args.output / "historical.json"
    write_json(path, manifest)
    try:
        references = [load_player(path, args.device) for path in args.references]
        hashes = [identity["checkpoint_hash"] for _, identity in references]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate reference checkpoint")
        manifest["references"] = [identity for _, identity in references]
        seen = set()
        for checkpoint in args.checkpoints:
            policy, identity = load_player(checkpoint, args.device)
            model_hash = identity["checkpoint_hash"]
            if model_hash in hashes or model_hash in seen:
                raise ValueError(
                    "Candidate checkpoints must be distinct from each other and references"
                )
            seen.add(model_hash)
            matches: list[dict[str, Any]] = []
            for reference, ref_identity in references:
                matches.extend(
                    {"reference_hash": ref_identity["checkpoint_hash"], **row}
                    for row in compare_policies(policy, reference)
                )
            manifest["models"].append({**identity, "matches": matches})
            write_json(path, manifest)
            print(
                f"{identity['run_id']}: {len(matches)} historical matchups completed",
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
