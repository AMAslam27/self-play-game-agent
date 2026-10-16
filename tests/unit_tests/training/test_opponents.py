import random
from pathlib import Path

import pytest
import torch

from models.tictactoe_q_network import TicTacToeQNetwork
from training.checkpoints import CHECKPOINT_VERSION, capture_rng_state
from training.config import resolve_config
from training.tictactoe.environment import TicTacToeAdapter
from training.tictactoe.opponents import TrainingOpponents


def pool_config(**settings):
    return resolve_config({
        "environment": {"opponent": "pool"},
        "network": {"hidden_sizes": [8, 8]},
        "opponent_pool": settings,
    }, Path("unused"))


def saved_model(path):
    config = pool_config()
    model = TicTacToeQNetwork(**config["network"])
    torch.save({"checkpoint_version": CHECKPOINT_VERSION, "config": config,
                "run_id": "source", "agent": {"online_network": model.state_dict()}}, path)
    return model


def assert_weights_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)


def test_snapshots_are_independent_frozen_and_preserve_training_rng():
    config = pool_config(snapshot_every_episodes=2, max_snapshots=3)
    learner = TicTacToeQNetwork(**config["network"])
    opponents = TrainingOpponents(config, torch.device("cpu"), random.Random(4))
    before = capture_rng_state()
    opponents.initialize(learner)
    torch.testing.assert_close(before["torch"], capture_rng_state()["torch"], rtol=0, atol=0)
    snapshot = opponents.policies["snapshot-0"].model
    assert snapshot is not learner
    assert learner.training and not snapshot.training
    assert all(not parameter.requires_grad for parameter in snapshot.parameters())
    original = {name: tensor.clone() for name, tensor in snapshot.state_dict().items()}
    with torch.no_grad():
        for parameter in learner.parameters():
            parameter.add_(1)
    assert_weights_equal(original, snapshot.state_dict())
    for episode in range(1, 9):
        opponents.after_episode(learner, episode)
    assert [entry["episode"] for entry in opponents.snapshots] == [0, 6, 8]
    assert set(opponents.policies) == {"snapshot-0", "snapshot-6", "snapshot-8"}


@pytest.mark.parametrize("category", ["random", "minimax", "latest", "historical"])
@pytest.mark.parametrize("seat", [1, -1])
def test_each_category_can_play_legal_complete_games_in_both_seats(category, seat):
    config = pool_config(snapshot_every_episodes=1)
    learner = TicTacToeQNetwork(**config["network"])
    opponents = TrainingOpponents(config, torch.device("cpu"), random.Random(4))
    opponents.initialize(learner)
    opponents.after_episode(learner, 1)
    # History now exists, so historical-only selection is available as well.
    config["opponent_pool"]["weights"] = {name: int(name == category) for name in (
        "random", "minimax", "latest", "historical")}
    selected = opponents.select()
    assert selected.category == category
    if category == "latest":
        assert selected.identity == "snapshot-1"
    if category == "historical":
        assert selected.identity == "snapshot-0"
    environment = TicTacToeAdapter(selected.policy, agent_player=seat)
    observation = environment.reset()
    done = False
    while not done:
        action = next(index for index, allowed in enumerate(observation.action_mask) if allowed)
        observation, reward, done = environment.step(action)
    assert reward in (-1, 0, 1)


def test_category_probabilities_do_not_depend_on_number_of_snapshots():
    config = pool_config()
    learner = TicTacToeQNetwork(**config["network"])
    first = TrainingOpponents(config, torch.device("cpu"), random.Random(2))
    second = TrainingOpponents(config, torch.device("cpu"), random.Random(2))
    for pool in (first, second):
        pool.initialize(learner)
        pool.add_snapshot(learner, 1)
    for episode in range(2, 6):
        second.add_snapshot(learner, episode)
    # Isolate the first-stage draw; choosing history consumes extra RNG draws.
    counts = []
    for pool in (first, second):
        categories = []
        for seed in range(300):
            pool.rng.seed(seed)
            categories.append(pool.select().category)
        counts.append(categories)
    assert counts[0] == counts[1]
    assert set(counts[0]) == {"random", "minimax", "latest", "historical"}


def test_empty_history_is_excluded_and_weights_are_renormalized():
    config = pool_config(weights={"random": 0, "minimax": 0, "latest": 0.1, "historical": 0.9})
    opponents = TrainingOpponents(config, torch.device("cpu"), random.Random(4))
    opponents.initialize(TicTacToeQNetwork(**config["network"]))
    assert all(opponents.select().category == "latest" for _ in range(30))


@pytest.mark.parametrize("mode", ["model", "pool"])
def test_imports_are_retained_and_restore_without_original_checkpoint(tmp_path, mode):
    path = tmp_path / "opponent.pt"
    original = saved_model(path)
    config = pool_config(max_snapshots=2)
    config["environment"]["opponent"] = mode
    config["opponent_model"]["checkpoint"] = str(path)
    config["opponent_pool"]["initial_checkpoints"] = [str(path)]
    pool = TrainingOpponents(config, torch.device("cpu"), random.Random(4))
    pool.initialize(TicTacToeQNetwork(**config["network"]))
    assert_weights_equal(original.state_dict(), pool.policies["imported-0"].model.state_dict())
    for episode in range(1, 5):
        pool.after_episode(original, episode * 1000)
    assert "imported-0" in pool.policies
    if mode == "model":
        assert pool.snapshots == []
        assert all(pool.select().identity == "imported-0" for _ in range(10))
    # Exercise weights_only-compatible serialization as well as restoration.
    state_file = tmp_path / "pool.pt"
    torch.save(pool.state_dict(), state_file)
    path.unlink()
    restored = TrainingOpponents(config, torch.device("cpu"), random.Random(999))
    before = capture_rng_state()
    restored.load_state_dict(torch.load(state_file, weights_only=True))
    torch.testing.assert_close(before["torch"], capture_rng_state()["torch"], rtol=0, atol=0)
    assert [pool.select().identity for _ in range(30)] == [restored.select().identity for _ in range(30)]
    for identity in pool.policies:
        assert_weights_equal(pool.policies[identity].model.state_dict(), restored.policies[identity].model.state_dict())
