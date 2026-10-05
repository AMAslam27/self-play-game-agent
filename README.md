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
same constructor settings under `network`. Config loading, action selection,
legal-action masking, and the learning loop will be added separately.
