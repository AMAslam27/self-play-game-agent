import copy
import unittest
from pathlib import Path

import torch
import yaml

from models.tictactoe import TicTacToeQNetwork
from training.dqn import DQNAgent
from training.utils import build_lr_scheduler, build_optimizer


def step_optimizer(optimizer, parameter):
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()


class TestOptimizerUtilities(unittest.TestCase):
    def test_supported_optimizers_receive_learning_rate_and_weight_decay(self):
        for name, expected in [
            ("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
        ]:
            with self.subTest(name=name):
                parameter = torch.nn.Parameter(torch.ones(1))
                config = {"name": name, "learning_rate": 0.02, "weight_decay": 0.1}
                optimizer = build_optimizer([parameter], config)
                self.assertIsInstance(optimizer, expected)
                self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)
                self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.1)

    def test_sgd_weight_decay_changes_weights_even_with_zero_loss_gradient(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        config = {"name": "sgd", "learning_rate": 0.1, "weight_decay": 0.2}
        optimizer = build_optimizer([parameter], config)
        step_optimizer(optimizer, parameter)
        torch.testing.assert_close(parameter, torch.tensor([0.98]))

    def test_optimizer_specific_options_are_passed_without_mutating_config(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        config = {"name": "sgd", "options": {"momentum": 0.9}}
        original = copy.deepcopy(config)
        optimizer = build_optimizer([parameter], config)
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.9)
        self.assertEqual(config, original)

    def test_invalid_optimizer_configs_fail(self):
        cases = [
            {"name": "unknown"}, {"name": None},
            {"learning_rate": 0}, {"learning_rate": float("nan")},
            {"learning_rate": True}, {"weight_decay": -0.1},
            {"weight_decay": float("inf")}, {"weight_decay": True},
            {"weight_deacy": 0.1}, {"options": None},
            {"options": {"lr": 0.1}}, {"options": {"weight_decay": 0.1}},
            {"options": {"unsupported": True}}, "adam",
        ]
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    build_optimizer([torch.nn.Parameter(torch.ones(1))], config)

    def test_disabled_scheduler(self):
        optimizer = build_optimizer([torch.nn.Parameter(torch.ones(1))])
        self.assertIsNone(build_lr_scheduler(optimizer))
        self.assertIsNone(build_lr_scheduler(optimizer, {"name": "none", "options": {}}))

    def test_step_scheduler_changes_lr_at_optimizer_update_boundaries(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = build_optimizer([parameter], {"learning_rate": 0.1})
        config = {"name": "step", "options": {"step_size": 2, "gamma": 0.5}}
        original = copy.deepcopy(config)
        scheduler = build_lr_scheduler(optimizer, config)
        rates = []
        for _ in range(4):
            step_optimizer(optimizer, parameter)
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertEqual(rates, [0.1, 0.05, 0.05, 0.025])
        self.assertEqual(config, original)

    def test_exponential_scheduler_changes_lr_every_update(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = build_optimizer([parameter], {"learning_rate": 0.1})
        scheduler = build_lr_scheduler(optimizer, {"name": "exponential", "options": {"gamma": 0.5}})
        step_optimizer(optimizer, parameter)
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)

    def test_scheduler_state_can_resume_the_same_schedule(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        config = {"name": "step", "options": {"step_size": 2, "gamma": 0.5}}
        optimizer = build_optimizer([parameter], {"learning_rate": 0.1})
        scheduler = build_lr_scheduler(optimizer, config)
        step_optimizer(optimizer, parameter)
        scheduler.step()

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        restored_optimizer = build_optimizer([restored_parameter], {"learning_rate": 0.1})
        restored_scheduler = build_lr_scheduler(restored_optimizer, config)
        restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        restored_scheduler.load_state_dict(copy.deepcopy(scheduler.state_dict()))
        for _ in range(3):
            step_optimizer(optimizer, parameter)
            scheduler.step()
            step_optimizer(restored_optimizer, restored_parameter)
            restored_scheduler.step()
            self.assertEqual(optimizer.param_groups[0]["lr"], restored_optimizer.param_groups[0]["lr"])

    def test_invalid_scheduler_configs_fail(self):
        optimizer = build_optimizer([torch.nn.Parameter(torch.ones(1))])
        cases = [
            {"name": "unknown"}, {"name": None},
            {"name": "none", "options": {"gamma": 0.5}},
            {"name": "step"},
            {"name": "step", "options": {"step_size": 0}},
            {"name": "step", "options": {"step_size": True}},
            {"name": "step", "options": {"step_size": 1.5}},
            {"name": "step", "options": {"step_size": 1, "extra": 1}},
            {"name": "exponential"},
            {"name": "exponential", "options": {"gamma": 0}},
            {"name": "exponential", "options": {"gamma": float("nan")}},
            {"name": "step", "options": {"step_size": 1, "optimizer": optimizer}},
            {"options": []}, {"extra": 1}, "step",
        ]
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    build_lr_scheduler(optimizer, config)

    def test_example_config_can_supply_agent_settings_directly(self):
        path = Path(__file__).resolve().parents[3] / "training/tictactoe/config.example.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        training = config["training"]
        agent = DQNAgent(
            TicTacToeQNetwork(**config["network"]),
            optimizer=training["optimizer"],
            lr_scheduler=training["lr_scheduler"],
            discount_factor=training["discount_factor"],
            target_update_every_updates=training["target_update_every_updates"],
            device="cpu",
        )
        self.assertEqual(agent.optimizer.param_groups[0]["lr"], 0.001)
        self.assertEqual(agent.optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertIsNone(agent.lr_scheduler)
