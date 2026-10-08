# Saved-model ranking

The local registry lives at `results/models.sqlite3`. It stores completed runs,
final checkpoint paths and SHA-256 hashes, evaluation protocols, and baseline
outcomes. Checkpoint and database files remain local experiment artifacts.

Register completed Tic-Tac-Toe DQN runs explicitly:

```sh
poetry run python -m evaluation.model_registry training/tictactoe/runs/<run-id>
```

Pass multiple run directories to import more than one run. Use `--registry`
for another database and `--evaluator-id` to distinguish opponent/evaluator
implementations. Reimport after extending a run or relocating its files.
Registration is transactional and repeatable. It imports only completed
evaluation batches at the final checkpoint's episode and update count.
Completed training runs now register their final checkpoints automatically.
Use `--registry <path>` to override the training registry. Resuming a run
deactivates its previously registered final checkpoint until it completes again.
If registration fails after successful training, the run remains completed,
the error is logged and recorded in `run.json`, and explicit import can be retried.
Runs with final checkpoint saving disabled can only register when their last
periodic checkpoint matches the final episode and update count.

## Ranking policy: tictactoe-baseline-v1

Only registered final checkpoints of completed runs are ranked. They need
complete greedy evaluation against random and minimax from both X and O.
Files must exist and match their registered hashes. Rank in this order:

1. Lowest maximum minimax loss rate across the two seats.
2. Highest minimum random-opponent return across the two seats.
3. Highest mean random-opponent return across the two seats.
4. Most recent run completion time, then run ID for a stable exact tie.

Return is `(wins - losses) / games`. Results from different protocols never
compete. If more than one eligible protocol exists, callers must specify its
fingerprint. Protocols include game, evaluator identity, encoding, selection,
opponent order (which affects seeds), seats, evaluation seed, games per batch,
and epsilon. Training hyperparameters and evaluation frequency are excluded.

Historical runs lack evaluator source fingerprints. The importer labels them
`legacy-tictactoe-baseline-v1`; this is an explicit compatibility assumption,
not verification that historical opponent implementations were identical.
Use distinct evaluator identities when importing runs with different evaluators.
The current deterministic minimax evaluation does not establish perfect play.
Exhaustive assessment will require a new ranking policy and protocol.

## Selection API

`players.checkpoint_selection.resolve_checkpoint` accepts:

- An explicit checkpoint path, including intermediate checkpoints. No registry
  is needed; the loader subsequently validates the checkpoint format.
- `latest`: the available final checkpoint of the most recently completed
  compatible run. Full baseline evaluation is not required.
- `best`: the first eligible checkpoint under the ranking above.

Automatic selection requires a connection from
`evaluation.model_registry.connect_registry`. It filters by game and model type.
Unavailable or changed checkpoint files are skipped; no eligible candidate
produces a clear error. `evaluation.tictactoe_ranking.rank_checkpoints` returns the ranked
records with experiment names, protocol IDs, component scores, and an explanation.
The first ranking policy supports Tic-Tac-Toe; other games need their own policy.
The runner accepts `--x dqn` or `--o dqn` with corresponding `--x-agent` or
`--o-agent` selectors (`latest` or `best`), or `--x-checkpoint`/`--o-checkpoint`
paths. Both seats can use different models. `--x-protocol`/`--o-protocol` select
a comparison protocol for `best`. Model paths and hashes are displayed before
play and recorded in the results database's `run_models` table. Existing
results databases gain that table without changing their old match records.
