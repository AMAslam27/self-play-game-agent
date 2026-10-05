"""Run reproducible Tic-Tac-Toe DQN experiments and resume saved runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import signal
import subprocess
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml

from games.tictactoe.rules import PLAYER_O, PLAYER_X
from models.tictactoe import TicTacToeQNetwork
from training.checkpoints import (
    checkpoint_payload,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from training.config import PROJECT_ROOT, load_config, resolve_config
from training.dqn import DQNAgent
from training.metrics import (
    MetricsRecorder,
    restore_metric_offsets,
    utc_now,
    write_json,
)
from training.plots import plot_training_metrics
from training.tictactoe.environment import Observation, TicTacToeAdapter
from training.tictactoe.evaluation import evaluate_agent, make_opponent
from training.utils import ReplayBuffer, Transition, linear_epsilon

logger = logging.getLogger(__name__)
DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.example.yaml"


@dataclass
class Progress:
    episodes: int = 0
    decisions: int = 0
    training_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    wall_seconds: float = 0.0
    last_evaluation_episode: int = -1


def _seed_training(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def _code_version() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
                text=True, check=True, timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        # Slim training containers may omit Git, but a mounted checkout still
        # exposes its current commit. Dirty status remains explicitly unknown.
        git_dir = PROJECT_ROOT / ".git"
        try:
            if git_dir.is_file():
                git_dir = (
                    PROJECT_ROOT / git_dir.read_text().strip().removeprefix("gitdir: ")
                ).resolve()
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref: "):
                ref = head.removeprefix("ref: ")
                ref_file = git_dir / ref
                if ref_file.exists():
                    head = ref_file.read_text().strip()
                else:
                    refs = (git_dir / "packed-refs").read_text().splitlines()
                    head = next(
                        line.split()[0] for line in refs if line.endswith(f" {ref}")
                    )
            return {"commit": head, "dirty": None, "note": "Git executable unavailable"}
        except (OSError, StopIteration):
            return {"commit": None, "dirty": None, "note": "Git metadata unavailable"}


def _save_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    if config["artifacts"]["save_resolved_config"]:
        temporary = run_dir / "config.yaml.tmp"
        temporary.write_text(
            yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
        )
        temporary.replace(run_dir / "config.yaml")


def run_training(
    config: Mapping[str, Any],
    *,
    resume: Path | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Train or resume at episode boundaries; return the run directory."""
    session_started = perf_counter()
    config = resolve_config(config, DEFAULT_RUNS_DIR)
    stop = should_stop if should_stop is not None else lambda: False
    experiment, training = config["experiment"], config["training"]
    artifacts = config["artifacts"]
    evaluation = config["evaluation"]
    exploration = config["exploration"]
    saved = load_checkpoint(resume) if resume is not None else None
    if saved is not None:
        # Only the episode target and device may change on resume.
        previous = resolve_config(saved["config"], DEFAULT_RUNS_DIR)
        previous["training"]["episodes"] = training["episodes"]
        previous["experiment"]["device"] = experiment["device"]
        if previous != config:
            raise ValueError(
                "Resume must use checkpoint settings; only episodes and device may change"
            )
    _seed_training(experiment["seed"], experiment["deterministic"])
    network = TicTacToeQNetwork(**config["network"])
    agent = DQNAgent(
        network,
        optimizer=training["optimizer"],
        lr_scheduler=training["lr_scheduler"],
        discount_factor=training["discount_factor"],
        target_update_every_updates=training["target_update_every_updates"],
        device=experiment["device"],
        rng=random.Random(experiment["seed"] + 2),
    )
    replay: ReplayBuffer[Observation] = ReplayBuffer(
        training["replay_capacity"], random.Random(experiment["seed"] + 3)
    )
    opponent_rng = random.Random(experiment["seed"] + 1)
    environment = TicTacToeAdapter(
        make_opponent(config["environment"]["opponent"], opponent_rng)
    )

    if saved is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{stamp}_{uuid4().hex[:8]}"
        run_dir = Path(artifacts["output_root"]) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        progress = Progress()
        metadata: dict[str, Any] = {
            "run_id": run_id,
            "name": experiment["name"],
            "started_at": utc_now(),
            "sessions": [],
        }
    else:
        assert resume is not None
        run_dir = resume.resolve().parent.parent
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        run_id = saved["run_id"]
        if metadata["run_id"] != run_id:
            raise ValueError("Checkpoint does not belong to this run directory")
        progress = Progress(**saved["progress"])
        if training["episodes"] < progress.episodes:
            raise ValueError("Episode target is below the checkpoint's completed episodes")
        restore_checkpoint(saved, agent, replay, opponent_rng, Observation)
        restore_metric_offsets(run_dir, saved["metric_offsets"])

    session = len(metadata["sessions"]) + 1
    metadata["sessions"].append(
        {
            "session": session,
            "started_at": utc_now(),
            "device": str(agent.device),
            "resume_checkpoint": str(resume) if resume is not None else None,
            "starting_episode": progress.episodes,
            "code_version": _code_version() if artifacts["save_code_version"] else None,
        }
    )
    metadata.update(
        {
            "config": config,
            "status": "running",
            "finished_at": None,
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
            },
        }
    )
    _save_config(run_dir, config)
    wall_base = max(
        progress.wall_seconds, float(metadata.get("progress", {}).get("wall_seconds", 0))
    )
    window_returns: deque[float] = deque(maxlen=artifacts["metrics_every_episodes"])
    log_handler = logging.FileHandler(run_dir / "training.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(log_handler)

    def update_metadata(status: str) -> None:
        progress.wall_seconds = wall_base + perf_counter() - session_started
        metadata.update({
            "status": status, "progress": asdict(progress), "learning_updates": agent.learning_updates,
            "episodes_per_second": progress.episodes / progress.training_seconds if progress.training_seconds else 0.0,
            "updates_per_second": agent.learning_updates / progress.training_seconds if progress.training_seconds else 0.0,
            "finished_at": utc_now() if status != "running" else None,
        })
        if status != "running":
            metadata["sessions"][-1].update({
                "finished_at": utc_now(), "status": status,
                "duration_seconds": perf_counter() - session_started,
            })
        write_json(run_dir / "run.json", metadata)

    try:
        with MetricsRecorder(run_dir) as metrics:
            def checkpoint() -> None:
                progress.wall_seconds = wall_base + perf_counter() - session_started
                payload = checkpoint_payload(
                    run_id=run_id, config=config, agent=agent, replay=replay,
                    opponent_rng=opponent_rng, progress=asdict(progress), metric_offsets=metrics.offsets(),
                )
                filename = (
                    f"episode-{progress.episodes:08d}_"
                    f"updates-{agent.learning_updates:08d}_session-{session:03d}.pt"
                )
                path = run_dir / "checkpoints" / filename
                save_checkpoint(path, payload)
                metadata["last_checkpoint"] = str(path)
                update_metadata("running")

            def evaluate() -> bool:
                started = perf_counter()
                try:
                    completed = evaluate_agent(
                        agent, evaluation, metrics, run_id=run_id, session=session,
                        episode=progress.episodes, decisions=progress.decisions, should_stop=stop,
                    )
                    if completed:
                        progress.last_evaluation_episode = progress.episodes
                    return completed
                finally:
                    progress.evaluation_seconds += perf_counter() - started

            update_metadata("running")
            logger.info("Run %s: %s, device=%s", run_id, run_dir, agent.device)
            if saved is None:
                # Keep initialization recoverable if baseline evaluation fails.
                checkpoint()
            if progress.last_evaluation_episode != progress.episodes and (
                progress.episodes == 0 or progress.episodes % evaluation["every_episodes"] == 0
            ):
                evaluate()
                checkpoint()

            while progress.episodes < training["episodes"] and not stop():
                episode = progress.episodes + 1
                seat = config["environment"]["agent_seat"]
                plays_x = seat == "x" or (seat == "alternate" and episode % 2 == 1)
                player = PLAYER_X if plays_x else PLAYER_O
                started = perf_counter()
                observation = environment.reset(player)
                decisions = progress.decisions
                start_updates = agent.learning_updates
                decision_rows, update_rows = [], []
                episode_return = 0.0
                done = False
                while not done:
                    epsilon = linear_epsilon(
                        decisions,
                        exploration["initial_epsilon"],
                        exploration["final_epsilon"],
                        exploration["decay_steps"],
                    )
                    action = agent.select_action(observation, epsilon)
                    next_observation, reward, done = environment.step(action)
                    decisions += 1
                    decision_rows.append({
                        "session": session, "episode": episode, "decision": decisions,
                        "learning_updates": agent.learning_updates, "epsilon": epsilon,
                        "learning_rate": agent.optimizer.param_groups[0]["lr"],
                    })
                    replay.add(
                        Transition(observation, action, reward, next_observation, done)
                    )
                    if (
                        decisions >= training["learning_starts"]
                        and len(replay) >= training["batch_size"]
                        and decisions % training["train_every_steps"] == 0
                    ):
                        for _ in range(training["gradient_updates_per_step"]):
                            lr_used = agent.optimizer.param_groups[0]["lr"]
                            loss = agent.learn(replay.sample(training["batch_size"]))
                            update_rows.append({
                                "session": session, "episode": episode, "decision": decisions,
                                "update": agent.learning_updates, "loss": loss,
                                "lr_used": lr_used, "lr_next": agent.optimizer.param_groups[0]["lr"],
                            })
                    observation = next_observation
                    episode_return += reward
                duration = perf_counter() - started
                episode_decisions = decisions - progress.decisions
                progress.episodes, progress.decisions = episode, decisions
                progress.training_seconds += duration
                for row in decision_rows:
                    metrics.write("decisions", row)
                for row in update_rows:
                    metrics.write("updates", row)
                metrics.write("episodes", {
                    "session": session, "episode": episode, "seat": "x" if player == PLAYER_X else "o",
                    "outcome": int(episode_return), "return": episode_return,
                    "decisions": episode_decisions, "updates": agent.learning_updates - start_updates,
                    "total_decisions": decisions, "total_updates": agent.learning_updates,
                    "duration_seconds": duration, "training_seconds": progress.training_seconds,
                })
                window_returns.append(episode_return)
                if episode % artifacts["metrics_every_episodes"] == 0:
                    metrics.flush()
                    update_metadata("running")
                    logger.info("Episode %s/%s | return %.3f | updates %s | epsilon %.4f | LR %.6g | elapsed %.1fs", episode, training["episodes"], sum(window_returns) / len(window_returns), agent.learning_updates, epsilon, agent.optimizer.param_groups[0]["lr"], progress.wall_seconds)
                if stop():
                    break
                if episode % evaluation["every_episodes"] == 0:
                    evaluate()
                if episode % artifacts["checkpoint_every_episodes"] == 0:
                    checkpoint()
                if episode % artifacts["plot_every_episodes"] == 0 and not stop():
                    metrics.flush()
                    plot_training_metrics(run_dir, artifacts["metrics_every_episodes"])

            status = "interrupted" if stop() else "completed"
            if status == "completed" and progress.last_evaluation_episode != progress.episodes:
                if not evaluate():
                    status = "interrupted"
            if status == "interrupted" or artifacts["save_final_checkpoint"]:
                checkpoint()
            metrics.flush()
            plot_training_metrics(run_dir, artifacts["metrics_every_episodes"])
            update_metadata(status)
            logger.info("%s: %s episodes, %s updates | training %.1fs | evaluation %.1fs | total %.1fs | %s", status, progress.episodes, agent.learning_updates, progress.training_seconds, progress.evaluation_seconds, progress.wall_seconds, run_dir)
    except BaseException as error:
        try:
            update_metadata("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
        except Exception:
            logger.exception("Could not preserve final run metadata")
        try:
            plot_training_metrics(run_dir, artifacts["metrics_every_episodes"])
        except Exception:
            logger.exception("Could not generate plots after interruption/failure")
        raise
    finally:
        logger.removeHandler(log_handler)
        log_handler.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="YAML settings (default: config.example.yaml)")
    source.add_argument("--resume", type=Path, help="Resume a checkpoint in its existing run directory")
    parser.add_argument("--episodes", type=int, help="Override the total episode target, including prior episodes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.resume is not None:
            config = resolve_config(load_checkpoint(args.resume)["config"], DEFAULT_RUNS_DIR, episodes=args.episodes, device=args.device)
        else:
            config = load_config(args.config or DEFAULT_CONFIG_PATH, DEFAULT_RUNS_DIR, episodes=args.episodes, device=args.device)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    requested = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal requested
        if requested:
            raise KeyboardInterrupt
        requested = True
        logger.info("Stop requested; finishing the current game and saving progress")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        run_training(config, resume=args.resume, should_stop=lambda: requested)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()
