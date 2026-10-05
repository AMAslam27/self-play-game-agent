# Self-Play Game Agent

## Project Overview

Train an agent to play games through reinforcement learning and
self-play rather than relying on a labelled dataset.

The project will investigate whether an agent can develop a competent
strategy through repeated interaction with the game environment.

Games will be: Tic Tac Toe, Connect-4, Chess.

## Goals

-   Implement a reliable game environment.
-   Establish simple non-learning opponents.
-   Train a reinforcement-learning agent.
-   Investigate self-play.
-   Evaluate agents against fixed opponents and previous versions.
-   Measure training stability and performance.
-   Use the GPU effectively during training.
-   Provide a simple interface for human-vs-agent play.

## Planned Approach

### 1. Environment

Implement:

-   Board representations.
-   Legal actions.
-   State transitions.
-   Terminal-state detection.
-   Reward structure.
-   Efficient batch/environment interaction.

### 2. Baseline Opponents

Start with:

-   Random player.
-   Simple heuristic player.
-   Minimax player where practical.

These provide meaningful reference points for evaluating the learned
agent.

### 3. Reinforcement Learning

Start with a relatively simple method such as DQN or PPO.

Then investigate a stronger self-play approach:

``` text
Current agent
      ↓
Self-play
      ↓
Game trajectories
      ↓
Training data
      ↓
Policy / value network
      ↓
Updated agent
      ↓
Self-play
```

A later extension could combine policy/value learning with Monte Carlo
Tree Search.

### 4. Evaluation

Track:

-   Win rate.
-   Draw rate.
-   Average return.
-   Performance against fixed opponents.
-   Performance against previous model versions.
-   Training stability.
-   Inference latency.

Potentially maintain an Elo-style rating system for agents.

### 5. Human Interface

Provide a simple way to play:

``` text
Human
  ↓
Games interface
  ↓
Trained agent
```

This could be a small web or desktop interface.

## Good Practices

-   Keep the environment deterministic where appropriate for testing.
-   Unit-test legal moves and terminal-state detection.
-   Separate environment, agent, training and evaluation code.
-   Evaluate against fixed opponents that are not changing during
    training.
-   Use independent evaluation games rather than training games.
-   Record random seeds and training configurations.
-   Save model checkpoints.
-   Monitor training instability.
-   Avoid judging progress from a single game.
-   Evaluate multiple random seeds where practical.
-   Keep visualisation and game UI separate from the training
    implementation.

## Success Criteria

The project is successful when:

1.  The environment passes comprehensive tests.
2.  A baseline RL agent can learn non-trivial behaviour.
3.  Self-play produces measurable improvement.
4.  The final agent consistently beats simple baseline opponents.
5.  Training and evaluation are reproducible.
6.  A user can play against the trained agent.

## Possible Extensions

-   AlphaZero-style policy/value training.
-   Monte Carlo Tree Search.
-   Curriculum learning.
-   Population-based self-play.
-   Model-vs-model tournaments.
-   GPU-optimised parallel environments.

## Local CI checks

With Poetry and GNU Make available on your PATH, run these commands from the
project root:

```text
make install
make ci
```

`make ci` runs the same checks as `.github/workflows/ci.yml`: Ruff lint,
Ruff format verification, mypy, and pytest. Checks run sequentially and stop
on the first failure. Dependencies only need reinstalling when they change.
Running `make` without a target also runs the CI checks.

Individual checks are available as `make lint`, `make format-check`,
`make typecheck`, and `make test`. Use `make format` to apply formatting.

On Windows, these commands require GNU Make (not Microsoft's `nmake`).
The checks use your Poetry environment; GitHub Actions currently uses Python
3.11 on Ubuntu, so using Python 3.11 locally gives a closer match.

## Recording results

Every batch of games is recorded in `results/games.sqlite3` under the project
root, independent of the working directory. Each human replay creates a new run.
Completed games are committed individually. Runs have UTC start/end timestamps
and a status: running, completed, abandoned, interrupted, or failed. A hard
process kill may leave a run marked running; committed games remain available.

```text
poetry run python runner.py --x random --o random --games 1000 --quiet --seed 42
poetry run python runner.py --x minimax --o random --games 100 --db-file results/comparison.sqlite3
```

`--seed` resets Python's random generator at the start of each batch; replaying
the same policies with the same seed reproduces their random choices. A seed is
only useful for policies that use this generator. A custom `--db-file` path is
relative to the current working directory unless absolute.

`evaluation/queries.sql` contains run summaries and opponent comparisons to
execute against the database. The recorder uses parameterised SQL, foreign keys,
and transactions; no database server or extra dependency is needed. Diagnostic
messages use Python logging on stderr. SQLite files are excluded from Git.

### Automatic charts

Each game batch automatically saves a chart under `results/tictactoe/plots/`, named with
its recorded UTC start timestamp (including microseconds) and database run ID,
for example `20261001T123025123456Z_run-42.png`. Human replays have separate
charts showing only that batch. Runs without completed games create no chart.
`--plot-file results/chart.png` overrides the path; subsequent human replay
batches append their run ID to that custom filename to preserve earlier charts.

## RL training environment adapter

`training.environment.TicTacToeAdapter` exposes agent decisions against an
existing opponent policy. It defaults to the random opponent; minimax and other
`policy(game) -> action` callables can also be passed in.

```python
import random

from games.tictactoe.rules import PLAYER_O
from training.environment import TicTacToeAdapter

random.seed(42)
environment = TicTacToeAdapter(agent_player=PLAYER_O)
observation = environment.reset()
done = False
while not done:
    legal = [i for i, allowed in enumerate(observation.action_mask) if allowed]
    action = random.choice(legal)  # replace with the learning agent's choice
    observation, reward, done = environment.step(action)
```

Observations are immutable snapshots containing nine board values and nine
boolean action-mask values in square order (indices 0-8). Own pieces are `+1`,
opponent pieces `-1`, and empty squares `0`. Each step includes the agent's move
and, if the game continues, the opponent's reply. Rewards are `+1` for an agent
win, `-1` for a loss, and `0` for a draw or ongoing play.

Reset plays the opponent's opening move when the agent is O.
`reset(agent_player=PLAYER_X)` or `reset(agent_player=PLAYER_O)` switches seats
for a new episode. Terminal observations have no legal actions, even if empty
squares remain; learning targets must use `done` to disable bootstrapping.
Call reset before the first step, after termination, or after an opponent error.
Invalid agent actions raise `ValueError` without advancing the game. Training
opponents must return legal moves and cannot quit or modify the game directly.
The default opponent uses Python's random generator, so seed it before a run.

## Configurable Q-network

`models.blocks.build_blocks` assembles hidden layers from equal-length
`hidden_sizes`, `block_types`, and `activations` lists. Each size is that block's
output width; its input width comes from the previous block.

- `dense`: linear transformation followed by an activation.
- `dropout`: a dense block followed by dropout using the shared
  `dropout_probability`; call `model.eval()` to disable dropout for evaluation.
- `residual`: two linear layers with an activation between them, adding the
  result to the input. Input and output widths must match; no projection or
  activation is applied to the skip addition.

Supported activations are `relu`, `tanh`, `gelu`, and `identity`.
`models.tictactoe.TicTacToeQNetwork` uses these blocks and adds a linear output
head with nine unrestricted Q-values. It accepts a floating-point board tensor
of shape `(9,)` or a batch of shape `(batch_size, 9)`.

```python
import torch

from models.tictactoe import TicTacToeQNetwork

model = TicTacToeQNetwork(
    hidden_sizes=[64, 64],
    block_types=["dense", "residual"],
    activations=["relu", "relu"],
)
q_values = model(torch.zeros(9))
```

The example config at `training/tictactoe/config.example.yaml` describes the
same constructor settings under `network`. The DQN agent below handles action
selection and legal-action masking; the training entry point below loads the
config and coordinates the full training loop.

## DQN agent

`training.dqn.DQNAgent` accepts a Q-network and provides `select_action` and
`learn`. Observations need flat numeric `board` features and an `action_mask`;
the agent does not import any game-specific rules.

```python
import random

import torch

from models.tictactoe import TicTacToeQNetwork
from training.dqn import DQNAgent

torch.manual_seed(42)
agent = DQNAgent(
    TicTacToeQNetwork(),
    optimizer={"name": "adam", "learning_rate": 0.001, "weight_decay": 0.0},
    lr_scheduler={"name": "none", "options": {}},
    discount_factor=0.99,
    target_update_every_updates=250,
    device="auto",
    rng=random.Random(42),
)
```

`agent.select_action(observation, epsilon=0.1)` explores with a random legal
move 10% of the time; otherwise it chooses the highest legal Q-value. Greedy
inference disables gradients and dropout. Use `epsilon=0.0` for evaluation.

`agent.learn(buffer.sample(batch_size))` converts replay values to float32
tensor batches on the selected device, performs one optimizer update using
Smooth L1 loss, and returns the mean loss. Ongoing transitions bootstrap from the
highest legal target-network value; terminal transitions use only their reward
and never evaluate their next observations. Target weights are copied after
every configured number of optimizer updates, not episodes or game moves.

`device="auto"` selects CUDA when available and CPU otherwise. Replay storage
stays in Python values. The agent uses an independent exploration RNG; the
training script owns configuration loading, epsilon decay, replay sampling,
evaluation, and checkpoints. Smooth L1 remains the loss function.

Optimizer and scheduler settings are mappings matching the example YAML:
`optimizer=config["training"]["optimizer"]` and
`lr_scheduler=config["training"]["lr_scheduler"]`. YAML loading belongs to the
training script; the agent receives settings rather than a file path.

`training.utils.build_optimizer` supports `adam`, `adamw`, and `sgd` through a
factory dictionary. Set `learning_rate` and `weight_decay` directly in the
optimizer mapping; optional `options` holds optimizer-specific arguments such
as `momentum` for SGD. Weight decay defaults to zero for every optimizer,
including AdamW.

`training.utils.build_lr_scheduler` supports `none`, `step`, and `exponential`.
For example, `{"name": "step", "options": {"step_size": 5000, "gamma": 0.5}}`
halves the learning rate after every 5,000 successful optimizer updates.
Exponential scheduling uses `{"name": "exponential", "options": {"gamma": 0.9999}}`
to multiply the rate after each update. Evaluation and action selection do not
advance the scheduler. Keep both optimizer and scheduler `state_dict()` values
in future checkpoints to resume the schedule.

## Train a Tic-Tac-Toe agent

```text
poetry run python -m training.tictactoe.train --config training/tictactoe/config.example.yaml
poetry run python -m training.tictactoe.train --episodes 100 --device cpu
```

The second command uses the example config with a shorter episode target.
Very short runs may stay entirely in replay warm-up and therefore have no
learning updates; reduce `training.learning_starts` in a separate config when
testing learning with a small episode count.
Training outputs live under `training/tictactoe/runs/<unique-run-id>/` by
default, separately from source files. Each run contains resolved `config.yaml`,
`run.json`, `training.log`, a SQLite evaluation database, checkpoints, four raw
CSV metric files, and seven PNG plots under `plots/`:

- `epsilon.png` and `learning_rate.png` track exploration and LR actually used.
- `loss.png` preserves raw losses and plots a rolling episode mean.
- `training_return.png` and `training_outcomes.png` show exploratory training results.
- `evaluation.png` shows greedy win/draw/loss rates for each opponent and seat.
- `time.png` shows cumulative environment/learning time and evaluation-match time.

`run.json` records training, evaluation, and total active wall-clock seconds,
throughput, UTC start/end timestamps, device, runtime versions, and code version
when Git metadata is available. Training time measures environment interaction
and learning; CSV writes, setup, checkpointing, and plotting contribute to total
wall time. On resume, paused time is excluded. CSVs contain one row per decision,
learning update, completed episode, or evaluation batch. Episodes commit their
metrics only after the game finishes; warm-up produces no loss rows.

Evaluation runs before training, periodically, and at normal completion. It
uses its own fixed seeds, disables exploration/learning, and preserves training
RNG states. Results are recorded in the run's `games.sqlite3` using the existing
recorder; policy labels include the training run ID and learning-update count.
Minimax decisions are cached without changing its policy. Incomplete evaluation
batches have an interrupted status and are excluded from performance plots.

Press Ctrl+C once to finish the current game, save progress, and generate plots.
A second Ctrl+C interrupts immediately; the last saved checkpoint remains
available. Unexpected failures mark the run failed and preserve completed CSV
records without saving a partially trained episode as a resumable boundary.

```text
poetry run python -m training.tictactoe.train --resume training/tictactoe/runs/<run-id>/checkpoints/latest.pt
poetry run python -m training.tictactoe.train --resume training/tictactoe/runs/<run-id>/checkpoints/latest.pt --episodes 30000
```

`--episodes` is the total target, including episodes already completed.
`--device` may also be changed on resume; all other settings come from the
checkpoint. Checkpoints include online/target networks, optimizer, scheduler,
replay data, independent/global RNG states, progress, and CSV boundaries.
When metrics extend beyond the selected checkpoint, they are backed up under
`recoveries/` before restoring the saved boundary. SQLite retains all evaluation
attempts as an audit trail. Reproducibility applies on the same code, software,
and device setup; moving between CPU and GPU may change numerical results.

`artifacts.metrics_every_episodes` controls progress/flush intervals and the
rolling plot window. `artifacts.plot_every_episodes` controls periodic chart
refreshes; final charts are always generated. Charts can be regenerated from
the CSVs by calling `training.plots.plot_training_metrics(run_dir)`.

Using the existing Docker service:

```text
docker compose run --rm --entrypoint python tictactoe -m training.tictactoe.train --config training/tictactoe/config.example.yaml
```

The project mount preserves run artifacts on the host. To run the new pipeline
tests without pytest, use `python -m unittest tests.unit_tests.training.test_training`
in an environment with the project dependencies installed.
