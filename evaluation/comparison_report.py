"""Publish comparable offline assessments, seed variation, and policy checks."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from training.metrics import write_json


def assessment_rows(assessments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject incomplete, duplicated, or incompatible evidence before pooling."""
    if not assessments:
        raise ValueError("At least one assessment is required")
    protocols = {assessment["protocol_id"] for assessment in assessments}
    if len(protocols) != 1:
        raise ValueError("Assessments must use the same evaluation protocol")
    seen_hashes = set()
    seen_seeds = set()
    rows = []
    for assessment in assessments:
        batches = assessment["batches"]
        if (
            assessment["status"] != "completed"
            or len(batches) != assessment["requested_batches"]
            or any(batch["status"] != "completed" for batch in batches)
        ):
            raise ValueError("Only fully completed assessments can be compared")
        model = assessment["model"]
        config = assessment["model_config"]
        method = config["environment"]["opponent"]
        seed = config["experiment"]["seed"]
        identity = model["checkpoint_hash"]
        if identity in seen_hashes or (method, seed) in seen_seeds:
            raise ValueError("Duplicate checkpoint or training seed within a method")
        seen_hashes.add(identity)
        seen_seeds.add((method, seed))
        for summary in assessment["summary"]:
            if summary["games"] <= 0:
                raise ValueError("Completed summaries must contain games")
            rows.append(
                {
                    "training_method": method,
                    "training_seed": seed,
                    "run_id": model["experiment_run_id"],
                    "checkpoint_hash": identity,
                    **{
                        key: summary.get(key)
                        for key in (
                            "opponent",
                            "seat",
                            "games",
                            "wins",
                            "draws",
                            "losses",
                            "loss_rate",
                            "loss_rate_ci_lower",
                            "loss_rate_ci_upper",
                            "loss_rate_ci_method",
                            "loss_rate_ci_reason",
                        )
                    },
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            row["training_method"],
            row["training_seed"],
            row["opponent"],
            row["seat"],
        ),
    )


def summarise_training_seeds(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each trained model gets equal weight; SD is across training seeds."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["training_method"], row["opponent"], row["seat"])].append(row)
    result = []
    for (method, opponent, seat), group in sorted(groups.items()):
        rates = [row["losses"] / row["games"] for row in group]
        result.append(
            {
                "training_method": method,
                "opponent": opponent,
                "seat": seat,
                "training_seeds": len(group),
                "mean_loss_rate": mean(rates),
                "sample_sd_loss_rate": stdev(rates) if len(rates) > 1 else None,
                "min_loss_rate": min(rates),
                "max_loss_rate": max(rates),
            }
        )
    return result


def attach_evidence(
    evidence: dict[str, Any], rows: Sequence[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    """Require exactly the assessed checkpoints and matching model metadata."""
    if evidence["status"] != "completed":
        raise ValueError(f"Incomplete {kind} evidence")
    identities = {row["checkpoint_hash"]: row for row in rows}
    models = evidence["models"]
    hashes = [model["checkpoint_hash"] for model in models]
    if len(set(hashes)) != len(hashes) or set(hashes) != set(identities):
        raise ValueError(f"{kind} checkpoints do not match the offline assessments")
    for model in models:
        original = identities[model["checkpoint_hash"]]
        if any(
            model[key] != original[key]
            for key in ("run_id", "training_method", "training_seed")
        ):
            raise ValueError(f"Conflicting {kind} model metadata")
    return models


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({row["training_method"] for row in rows})
    opponents = sorted({row["opponent"] for row in rows})
    seats = [seat for seat in ("x", "o") if any(row["seat"] == seat for row in rows)]
    figure, axes = plt.subplots(
        len(opponents),
        len(seats),
        figsize=(4.5 * len(seats), 3.5 * len(opponents)),
        squeeze=False,
    )
    for i, opponent in enumerate(opponents):
        for j, seat in enumerate(seats):
            axis = axes[i, j]
            for position, method in enumerate(methods):
                group = [
                    row
                    for row in rows
                    if row["opponent"] == opponent
                    and row["seat"] == seat
                    and row["training_method"] == method
                ]
                values = [100 * row["losses"] / row["games"] for row in group]
                if not values:
                    continue
                offsets = [
                    position + (index - (len(group) - 1) / 2) * 0.05
                    for index in range(len(group))
                ]
                axis.scatter(
                    offsets,
                    values,
                    alpha=0.8,
                    label="Training seed" if position == 0 else None,
                )
                axis.scatter(
                    [position],
                    [mean(values)],
                    marker="_",
                    s=500,
                    color="black",
                    label="Mean" if position == 0 else None,
                )
            axis.set_xticks(range(len(methods)), methods)
            axis.set_title(f"{opponent.title()} opponent, agent {seat.upper()}")
            axis.set_ylabel("Loss rate (%)")
            maximum = max(
                100 * row["losses"] / row["games"]
                for row in rows
                if row["opponent"] == opponent and row["seat"] == seat
            )
            upper = max(5, maximum * 1.2) if opponent == "random" else 100
            axis.set_ylim(-upper * 0.03, upper * 1.03)
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle(
        "Each point is one trained model; bars show means across training seeds"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assessments",
        type=Path,
        nargs="+",
        required=True,
        help="assessment.json files",
    )
    parser.add_argument("--historical", type=Path, help="historical.json")
    parser.add_argument("--exhaustive", type=Path, help="exhaustive.json")
    parser.add_argument(
        "--output", type=Path, required=True, help="New report directory"
    )
    args = parser.parse_args()
    assessments = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.assessments
    ]
    rows = assessment_rows(assessments)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": assessments[0]["protocol_id"],
        "uncertainty": "Per-model Wilson intervals describe random-game sampling; sample SD describes training-seed variation. Neither applies to deterministic repeated matchups.",
        "baseline_summary": summarise_training_seeds(rows),
    }
    for kind in ("historical", "exhaustive"):
        source = getattr(args, kind)
        if source:
            evidence = json.loads(source.read_text(encoding="utf-8"))
            summary[kind] = attach_evidence(evidence, rows, kind)
            if kind == "historical":
                summary["references"] = evidence["references"]
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "assessments"
    inputs.mkdir()
    for assessment, source in zip(assessments, args.assessments, strict=True):
        # Use the validated hash rather than untrusted manifest paths as a filename.
        identity = assessment["model"]["checkpoint_hash"]
        if len(identity) != 64 or any(
            char not in "0123456789abcdef" for char in identity
        ):
            raise ValueError("Invalid checkpoint hash")
        shutil.copyfile(source, inputs / f"{identity}.json")
    for kind in ("historical", "exhaustive"):
        if source := getattr(args, kind):
            shutil.copyfile(source, args.output / f"{kind}.json")
    write_csv(args.output / "models.csv", rows)
    write_csv(args.output / "training-seeds.csv", summary["baseline_summary"])
    for kind, field in (("historical", "matches"), ("exhaustive", "seats")):
        if kind in summary:
            detailed = []
            for model in summary[kind]:
                for result in model[field]:
                    detail = {
                        key: model[key]
                        for key in (
                            "run_id",
                            "checkpoint_hash",
                            "training_method",
                            "training_seed",
                        )
                    }
                    detail.update(result)
                    for key in ("moves", "losing_sequence"):
                        if key in detail:
                            detail[key] = json.dumps(detail[key])
                    detailed.append(detail)
            write_csv(args.output / f"{kind}.csv", detailed)
    write_json(args.output / "summary.json", summary)
    plot_results(rows, args.output / "baseline-losses.png")
    print(f"Published {len(assessments)} models to {args.output}")


if __name__ == "__main__":
    main()
