# Offline baseline evaluation

Evaluate a saved Tic-Tac-Toe model independently of training:

```sh
poetry run python -m evaluation.offline_eval --checkpoint <checkpoint.pt> --seeds 12345 23456 34567 --games 1000 --device cpu
```

The model is loaded once and plays greedy legal moves. No learning, exploration,
training resume, or modification of the checkpoint/training artifacts occurs.
The same match evaluator is used by training. Its independent opponent RNGs
and seed offsets retain the existing training evaluation behaviour.

`--games` is the number of games **per opponent, agent seat, and base seed**.
The defaults are random and minimax opponents, both seats, base seed 12345,
and 1,000 games per combination. Two opponents, two seats, and three seeds
therefore request 12,000 games. Use `--opponents random` or `--seats o` to narrow
the assessment. Explicit seeds must be distinct integers in `[0, 2**32)`.

Use `--agent latest` or `--agent best` instead of `--checkpoint` to resolve a
registered final checkpoint. `--registry <path>` chooses another registry.
If `best` needs a comparison protocol, pass `--protocol <registry-protocol-id>`.
This selects the model; the new assessment receives its own protocol fingerprint.

## Docker

Run inside the same environment where the model registry was populated:

```sh
docker compose run --rm --entrypoint python tictactoe -m evaluation.offline_eval --agent best --seeds 12345 23456 34567 --games 1000
```

An explicit `--checkpoint` path bypasses registry selection. The Compose mount
makes assessment files available in the local project directory.

## Assessment artifacts

Each invocation creates a unique directory under
`results/tictactoe/evaluations/`. Override the parent using `--output-root`.
Previous assessments are preserved, and these generated directories are ignored
by Git. Directory names separate the UTC date, time, microseconds, and random
suffix, for example `2026-10-15_15-02-03_721656_UTC_a7451141`.
Each assessment contains:

- `assessment.json`: assessment ID and status, checkpoint identity/hash, saved
  model config, runtime versions, evaluation protocol/fingerprint, per-batch
  outcomes, and aggregate summaries for each opponent and seat.
- `evaluation.csv`: outcomes, return, duration, base/derived seeds, and recording
  IDs for each opponent/seat/seed batch.
- `games.sqlite3`: individual game outcomes and exact model identities, using
  the existing match-recording schema.

Seed offsets are `base_seed + opponent_index * 2 + (seat == 'o')`. Opponent order
is therefore part of the protocol. Every batch uses a fresh private RNG, and
the caller's global Python, NumPy, and PyTorch RNG state is preserved.

Console and JSON summaries pool only completed batches. Partial batches remain
in the CSV/JSON audit records and are excluded from the aggregate. All completed
games are committed individually to SQLite. A first Ctrl+C finishes the current
game, records the partial batch, and exits with status 130. A second Ctrl+C or
a policy error can interrupt a batch before its CSV row is written; already
committed games remain available in SQLite. Failed assessments record the error
in their manifest and exit with status 1. Successful assessments exit with 0.

Offline assessments do not modify the model registry's ranking evidence or
change which checkpoint is `best`. The separate [policy assessment commands](policy-assessment.md)
provide exhaustive checks and comparison reporting without registry integration.
Repeated games against the current
deterministic minimax policy are sampled evaluation, not proof of perfect play.

## Loss-rate uncertainty

New assessments use manifest schema version 2 and report two-sided **95% Wilson
confidence intervals** for the loss rate against the random opponent. Intervals
are calculated per completed seed batch and again from the pooled counts for
each opponent and seat; interval endpoints are not averaged across seeds.
The console prints percentages, for example:

```text
loss 1.10% (95% CI 0.78%-1.54%)
loss 0.00% (95% CI N/A: deterministic opponent)
```

JSON batch/summary records and batch CSV rows add `loss_rate`,
`loss_rate_ci_lower`, `loss_rate_ci_upper`, `loss_rate_ci_method`,
`loss_rate_ci_confidence`, and `loss_rate_ci_reason`. Numeric rates and bounds
are proportions in `[0, 1]`; confidence is `0.95` and method is `wilson` where
applicable. Unavailable intervals use JSON nulls (blank numeric CSV fields)
and a reason: `deterministic_opponent`, `incomplete_batch`, or
`no_completed_games`. The manifest's `uncertainty` section records the method,
confidence level, assumptions, and scope. Earlier assessment files are preserved.

These intervals quantify sampling uncertainty for one fixed model, assuming
independent games with a fixed loss probability. They do not estimate variation
between training seeds. Minimax and the greedy agent are both deterministic;
repeated identical matchups do not support a binomial uncertainty estimate for
unexplored opponent move sequences. Partial batches are excluded from intervals
and pooled summaries. Zero sampled losses against random play still produce
a positive upper confidence bound.

The [NIST statistical handbook](https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
describes the Wilson method. The match protocol, ranking rules, and CLI options
are unchanged; rerun the same command to generate a new assessment with intervals.
