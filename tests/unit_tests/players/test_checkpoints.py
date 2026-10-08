import random

import pytest
import torch
from torch import nn

from models.tictactoe import TicTacToeQNetwork
from players.checkpoints import load_model_checkpoint
from training.checkpoints import CHECKPOINT_VERSION, checkpoint_payload
from training.dqn import DQNAgent
from training.utils import ReplayBuffer


def write_checkpoint(tmp_path, config, model):
    path = tmp_path / "agent.pt"
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_id": "test-run",
        "config": config,
        "agent": {"online_network": model.state_dict()},
        # Invalid training state must never be restored by inference loading.
        "rng": "unused", "replay": "unused", "optimizer": "unused",
    }
    torch.save(payload, path)
    return path, payload


@pytest.mark.parametrize("architecture", ["linear", "tictactoe"])
def test_loads_different_models_and_preserves_rng(tmp_path, architecture):
    config = {"architecture": architecture, "network": {"hidden_sizes": [8],
              "block_types": ["dense"], "activations": ["relu"]}}

    def factory(saved):
        if saved["architecture"] == "linear":
            return nn.Linear(9, 9)
        return TicTacToeQNetwork(**saved["network"])

    original = factory(config)
    path, _ = write_checkpoint(tmp_path, config, original)
    rng = torch.get_rng_state().clone()
    loaded = load_model_checkpoint(str(path), factory, device="cpu")
    assert torch.equal(torch.get_rng_state(), rng)
    assert loaded.run_id == "test-run"
    assert loaded.config == config
    assert loaded.checkpoint_path == path.resolve()
    assert not loaded.model.training
    assert next(loaded.model.parameters()).device.type == "cpu"
    inputs = torch.arange(9, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(inputs), original(inputs))


def test_loads_real_training_payload_using_only_online_weights(tmp_path):
    config = {"network": {"hidden_sizes": [8], "block_types": ["dense"],
                          "activations": ["relu"]}}
    agent = DQNAgent(TicTacToeQNetwork(**config["network"]), device="cpu")
    # Distinguish the online model from its target model.
    with torch.no_grad():
        agent.online_network.output.bias.fill_(3.0)
    payload = checkpoint_payload(run_id="real-run", config=config, agent=agent,
                                 replay=ReplayBuffer(5), opponent_rng=random.Random(1),
                                 progress={}, metric_offsets={})
    path = tmp_path / "real.pt"
    torch.save(payload, path)
    loaded = load_model_checkpoint(path, lambda saved: TicTacToeQNetwork(**saved["network"]), "cpu")
    assert torch.equal(loaded.model.output.bias, agent.online_network.output.bias)
    assert not torch.equal(loaded.model.output.bias, agent.target_network.output.bias)


@pytest.mark.parametrize("field,value,message", [
    ("checkpoint_version", -1, "Unsupported"),
    ("config", [], "configuration"),
    ("run_id", "", "run ID"),
    ("agent", {}, "online-network"),
])
def test_rejects_invalid_metadata_before_factory(tmp_path, field, value, message):
    path, payload = write_checkpoint(tmp_path, {}, nn.Linear(2, 2))
    payload[field] = value
    torch.save(payload, path)

    def factory(config):
        pytest.fail("Invalid metadata should be rejected before construction")

    with pytest.raises(ValueError, match=message):
        load_model_checkpoint(path, factory, "cpu")


def test_rejects_incompatible_weights_and_preserves_rng_on_failure(tmp_path):
    path, _ = write_checkpoint(tmp_path, {}, nn.Linear(2, 2))
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="incompatible"):
        load_model_checkpoint(path, lambda config: nn.Linear(3, 3), "cpu")
    assert torch.equal(torch.get_rng_state(), rng)


def test_factory_cannot_mutate_returned_configuration(tmp_path):
    config = {"network": {"width": 2}}
    path, _ = write_checkpoint(tmp_path, config, nn.Linear(2, 2))

    def factory(saved):
        saved["network"]["width"] = 99
        return nn.Linear(2, 2)

    assert load_model_checkpoint(path, factory, "cpu").config == config


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(tmp_path / "missing.pt", lambda config: nn.Linear(2, 2), "cpu")
