"""Regenerate training figures from raw CSVs without rerunning training."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from training.metrics import read_metrics


def _rolling(values: list[float], window: int) -> list[float]:
    recent: deque[float] = deque()
    total = 0.0
    result = []
    for value in values:
        recent.append(value)
        total += value
        if len(recent) > window:
            total -= recent.popleft()
        result.append(total / len(recent))
    return result


def _save(fig: Figure, path: Path) -> None:
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    finally:
        fig.clear()


def _figure(title: str, xlabel: str, ylabel: str):
    fig = Figure(figsize=(8, 4.5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.2)
    return fig, ax


def plot_training_metrics(run_dir: Path, window_episodes: int = 100) -> list[Path]:
    """Save LR, epsilon, loss, return/outcome, evaluation, and timing curves."""
    if window_episodes <= 0:
        raise ValueError("Plot window must be positive")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    decisions = read_metrics(run_dir, "decisions")
    updates = read_metrics(run_dir, "updates")
    episodes = read_metrics(run_dir, "episodes")
    evaluation = [
        row
        for row in read_metrics(run_dir, "evaluation")
        if row["status"] == "completed"
    ]
    saved = []

    fig, ax = _figure("Exploration", "Agent decisions", "Epsilon used")
    ax.plot(
        [int(row["decision"]) for row in decisions],
        [float(row["epsilon"]) for row in decisions],
    )
    ax.set_ylim(0, 1.05)
    path = plots_dir / "epsilon.png"
    _save(fig, path)
    saved.append(path)

    fig, ax = _figure("Learning rate", "Learning updates", "Learning rate used")
    if updates:
        ax.step(
            [int(row["update"]) for row in updates],
            [float(row["lr_used"]) for row in updates],
            where="post",
        )
    elif decisions:
        ax.plot([0], [float(decisions[0]["learning_rate"])], "o")
    else:
        ax.text(
            0.5, 0.5, "No learning updates yet", transform=ax.transAxes, ha="center"
        )
    path = plots_dir / "learning_rate.png"
    _save(fig, path)
    saved.append(path)

    fig, ax = _figure("Training loss", "Learning updates", "Smooth L1 loss")
    if updates:
        x = [int(row["update"]) for row in updates]
        losses = [float(row["loss"]) for row in updates]
        ax.plot(x, losses, alpha=0.2, label="Raw loss")
        # Aggregate by episode before smoothing: one episode may have many updates.
        grouped: dict[int, tuple[int, list[float]]] = {}
        for row in updates:
            episode = int(row["episode"])
            if episode not in grouped:
                grouped[episode] = (int(row["update"]), [])
            grouped[episode][1].append(float(row["loss"]))
            grouped[episode] = (int(row["update"]), grouped[episode][1])
        ax.plot(
            [value[0] for value in grouped.values()],
            _rolling(
                [sum(value[1]) / len(value[1]) for value in grouped.values()],
                window_episodes,
            ),
            label=f"Rolling episode mean (window {window_episodes})",
        )
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "Warm-up: no loss recorded", transform=ax.transAxes, ha="center"
        )
    path = plots_dir / "loss.png"
    _save(fig, path)
    saved.append(path)

    fig, ax = _figure(
        "Training return", "Completed episodes", "Undiscounted episode return"
    )
    x = [int(row["episode"]) for row in episodes]
    returns = [float(row["return"]) for row in episodes]
    ax.scatter(x, returns, alpha=0.12, s=8, label="Raw return")
    ax.plot(
        x,
        _rolling(returns, window_episodes),
        label=f"Rolling mean ({window_episodes} episodes)",
    )
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    path = plots_dir / "training_return.png"
    _save(fig, path)
    saved.append(path)

    fig = Figure(figsize=(9, 7))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)
    for seat, ax in zip(("x", "o"), axes, strict=True):
        rows = [row for row in episodes if row["seat"] == seat]
        for outcome, label in ((1, "Win"), (0, "Draw"), (-1, "Loss")):
            ax.plot(
                [int(row["episode"]) for row in rows],
                _rolling(
                    [float(int(row["outcome"]) == outcome) for row in rows],
                    window_episodes,
                ),
                label=label,
            )
        ax.set(
            title=f"Training as {seat.upper()}",
            xlabel="Completed episodes",
            ylabel="Rolling outcome rate",
            ylim=(0, 1.05),
        )
        ax.legend()
    path = plots_dir / "training_outcomes.png"
    _save(fig, path)
    saved.append(path)

    combinations = sorted({(row["opponent"], row["seat"]) for row in evaluation})
    fig = Figure(figsize=(9, max(3, 3 * len(combinations))))
    FigureCanvasAgg(fig)
    axes = fig.subplots(max(1, len(combinations)), 1, squeeze=False)
    for (opponent, seat), ax in zip(combinations, axes[:, 0], strict=False):
        rows = [
            row
            for row in evaluation
            if row["opponent"] == opponent and row["seat"] == seat
        ]
        for column, label in (("wins", "Win"), ("draws", "Draw"), ("losses", "Loss")):
            ax.plot(
                [int(row["episode"]) for row in rows],
                [int(row[column]) / int(row["games"]) for row in rows],
                marker="o",
                label=label,
            )
        ax.set(
            title=f"Greedy evaluation vs {opponent}, agent {seat.upper()}",
            xlabel="Completed training episodes",
            ylabel="Outcome rate",
            ylim=(0, 1.05),
        )
        ax.legend()
    if not combinations:
        axes[0, 0].text(0.5, 0.5, "No completed evaluation batches", ha="center")
    path = plots_dir / "evaluation.png"
    _save(fig, path)
    saved.append(path)

    fig, ax = _figure("Training time", "Completed episodes", "Cumulative seconds")
    ax.plot(
        x,
        [float(row["training_seconds"]) for row in episodes],
        label="Environment and learning",
    )
    elapsed = 0.0
    eval_x, eval_y = [], []
    for row in read_metrics(run_dir, "evaluation"):
        elapsed += float(row["duration_seconds"])
        eval_x.append(int(row["episode"]))
        eval_y.append(elapsed)
    ax.plot(eval_x, eval_y, label="Evaluation matches")
    ax.legend()
    path = plots_dir / "time.png"
    _save(fig, path)
    saved.append(path)
    return saved
