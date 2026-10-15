"""Create aggregate match charts independently of the game runner."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_results(results, policy_x_name, policy_o_name, *, output_path):
    """Plot X wins, draws and O wins from a {-1, 0, 1} result mapping.

    Save the figure to output_path without opening a window.
    Return the saved Path, or None when no games completed.
    """
    counts = [results[1], results[0], results[-1]]
    if any(count < 0 for count in counts):
        raise ValueError("Result counts cannot be negative")
    total = sum(counts)
    if total == 0:
        return None

    percentages = [100 * count / total for count in counts]
    labels = [f"X wins\n({policy_x_name})", "Draws", f"O wins\n({policy_o_name})"]
    fig = Figure(figsize=(7, 5))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    try:
        bars = ax.bar(labels, percentages, color=["#4477AA", "#999999", "#EE7733"])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Percentage of completed games")
        ax.set_title(f"{policy_x_name} vs {policy_o_name} — {total} games")
        ax.bar_label(
            bars,
            labels=[
                f"{count} ({percentage:.1f}%)"
                for count, percentage in zip(counts, percentages, strict=True)
            ],
            padding=4,
        )
        fig.tight_layout()

        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=150)
        return saved_path
    finally:
        fig.clear()
