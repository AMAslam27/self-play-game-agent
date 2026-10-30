# Tic-Tac-Toe policy assessment and comparison

These commands assess existing checkpoints without training or changing the
model registry. All output directories must be new; existing evidence is never
overwritten. Use explicit checkpoint paths to keep model selection fixed.

## Exhaustive opponent responses

```sh
poetry run python -m evaluation.exhaustive --checkpoints <model-a.pt> <model-b.pt> --device cuda --output results/tictactoe/exhaustive/my-assessment
```

For each checkpoint and each seat, the evaluator starts from the empty board.
It follows the model's legal greedy move and recursively visits every legal
opponent reply. It stops at the first win or full board. Greedy ties use the
lowest square index, matching saved-agent play. Model actions are cached by
board and turn; the policy must be deterministic and depend only on that state.

`exhaustive.json` records checkpoint paths, SHA-256 hashes, training seeds,
runtime versions, and per-seat results. `unbeatable: true` means no opponent
continuation beats this exact policy from the empty board on the recorded
device. It does not establish optimal actions on unreachable boards, or that
the model exploits every opponent mistake. Different devices/software can
change floating-point action selection; use the recorded runtime to reproduce.

`wins`, `draws`, and `losses` count terminal **move sequences**. They are not
probabilities under random play: opponent choices have different branching
factors and games have different lengths. Transposed histories count separately;
`states` counts distinct visited board/turn pairs. `losing_sequence` gives one
complete counterexample when exploitable. Moves are zero-based square indices,
row major (0–8), starting with X and alternating seats. It is not necessarily the
shortest counterexample.

## Fixed historical references

```sh
poetry run python -m evaluation.historical --checkpoints <model-a.pt> <model-b.pt> --references <old-random.pt> <old-pool.pt> --device cuda --output results/tictactoe/historical/my-assessment
```

References load once and stay frozen. Each candidate plays one greedy game per
reference and seat. `historical.json` includes both participants' identities,
the move sequence, and outcomes from the candidate's perspective. Repeated games
or extra seeds would duplicate these deterministic matchups, so no sampling
confidence interval is reported. Results characterize this reference set only;
references may share training settings/seeds with candidates. Duplicate models
and candidate/reference overlap by checkpoint hash are rejected.

Both commands save progress after each model. Errors and Ctrl+C mark manifests
failed/interrupted and preserve completed models. Such manifests cannot be used
as complete comparison evidence; rerun into a new directory.

## Publish a comparison

```sh
poetry run python -m evaluation.comparison_report --assessments <assessment-a.json> <assessment-b.json> --historical <historical.json> --exhaustive <exhaustive.json> --output docs/results/my-comparison
```

The report requires completed offline assessments with the same protocol ID.
It rejects duplicate checkpoints and duplicate training seeds within a method.
Optional historical/exhaustive manifests must be complete and identify exactly
the same candidates. The report creates:

- `models.csv`: each model's per-opponent/seat outcomes and existing Wilson intervals.
- `training-seeds.csv`: equal-weight model means, sample standard deviations,
  minima, maxima, and numbers of training seeds.
- `baseline-losses.png`: individual trained-model loss rates and method means.
- `historical.csv` and `exhaustive.csv` when their manifests are supplied.
- `summary.json`, original assessment JSON files, and optional source manifests.

Sample standard deviations describe variation across independently trained
models. Wilson intervals describe random-game sampling for one fixed model.
The method mean is not weighted by evaluation game count, and neither terminal
path counts nor deterministic matchups receive binomial intervals. A single
training seed has no estimated sample standard deviation. The report does not
claim statistical significance or change registry rankings.

See the [completed comparison](results/2026-10-30-comparison/README.md) for the
published ten-model experiment and its exact reproduction commands.
