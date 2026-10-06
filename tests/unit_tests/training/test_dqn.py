import random
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch import nn

from models.tictactoe import TicTacToeQNetwork
from training.dqn import DQNAgent
from training.tictactoe.environment import TicTacToeAdapter
from training.utils import ReplayBuffer, Transition


@dataclass(frozen=True)
class Features:
    board: tuple[float, ...] = (0.0, 0.0)
    action_mask: tuple[bool, ...] = (True, False, True)


def constant_network():
    network = nn.Linear(2, 3)
    with torch.no_grad():
        network.weight.zero_()
        network.bias.copy_(torch.tensor([0.2, 100.0, 0.8]))
    return network


def make_agent(**kwargs):
    return DQNAgent(constant_network(), device="cpu", **kwargs)


def terminal_transition():
    return Transition(Features(), 0, 1.0, Features(action_mask=(False,) * 3), True)


class TestDQNAgent(unittest.TestCase):
    def test_greedy_action_masks_highest_illegal_value(self):
        agent = make_agent()
        self.assertEqual(agent.select_action(Features()), 2)

    def test_greedy_ties_choose_first_legal_index(self):
        agent = make_agent()
        with torch.no_grad():
            agent.online_network.bias.fill_(1.0)
        self.assertEqual(agent.select_action(Features()), 0)

    def test_exploration_uses_only_legal_actions_without_network_inference(self):
        agent = make_agent(rng=random.Random(42))
        with patch.object(agent.online_network, "forward", side_effect=AssertionError):
            actions = [agent.select_action(Features(), epsilon=1.0) for _ in range(50)]
        self.assertEqual(set(actions), {0, 2})

    def test_exploration_is_reproducible_and_independent_of_global_random(self):
        first = make_agent(rng=random.Random(42))
        second = make_agent(rng=random.Random(42))
        state = random.getstate()
        self.assertEqual(
            [first.select_action(Features(), 0.5) for _ in range(30)],
            [second.select_action(Features(), 0.5) for _ in range(30)],
        )
        self.assertEqual(random.getstate(), state)

    def test_greedy_inference_disables_gradients_and_dropout_then_restores_mode(self):
        network = nn.Sequential(nn.Dropout(0.5), constant_network())
        agent = DQNAgent(network, device="cpu")
        observations = []

        def record_mode(module, inputs):
            observations.append((module.training, module[0].training, torch.is_grad_enabled()))

        hook = network.register_forward_pre_hook(record_mode)
        try:
            for training in (True, False):
                network.train(training)
                self.assertEqual(agent.select_action(Features()), 2)
                self.assertEqual(network.training, training)
                self.assertEqual(network[0].training, training)
        finally:
            hook.remove()
        self.assertEqual(observations, [(False, False, False)] * 2)

    def test_invalid_epsilon_and_terminal_action_selection_fail(self):
        agent = make_agent()
        for epsilon in (-0.1, 1.1, float("nan")):
            with self.subTest(epsilon=epsilon):
                with self.assertRaisesRegex(ValueError, "epsilon"):
                    agent.select_action(Features(), epsilon)
        with self.assertRaisesRegex(ValueError, "legal moves"):
            agent.select_action(Features(action_mask=(False,) * 3))

    def test_terminal_batch_never_calls_target_network(self):
        agent = make_agent()
        with patch.object(agent.target_network, "forward", side_effect=AssertionError):
            loss = agent.learn([terminal_transition()])
        # Smooth L1 in its quadratic region: 0.5 * (1.0 - 0.2)^2.
        self.assertAlmostEqual(loss, 0.32, places=6)

    def test_nonterminal_target_uses_frozen_network_and_masks_illegal_values(self):
        agent = make_agent(discount_factor=0.5)
        # Change only online values, proving bootstrapping uses the target copy.
        with torch.no_grad():
            agent.online_network.bias[2] = 5.0
        loss = agent.learn([Transition(Features(), 0, 0.1, Features(), False)])
        # Target = 0.1 + 0.5 * 0.8 = 0.5, not the illegal 100 or online 5.
        self.assertAlmostEqual(loss, 0.5 * (0.5 - 0.2) ** 2, places=6)

    def test_mixed_batch_has_finite_mean_loss(self):
        agent = make_agent(discount_factor=0.5)
        loss = agent.learn([
            terminal_transition(),
            Transition(Features(), 0, 0.1, Features(), False),
        ])
        self.assertAlmostEqual(loss, (0.32 + 0.045) / 2, places=6)

    def test_learning_enables_online_dropout_but_keeps_target_inference_frozen(self):
        agent = DQNAgent(
            nn.Sequential(nn.Dropout(0.5), constant_network()), device="cpu"
        )
        agent.online_network.eval()
        modes = []

        def record_mode(module, inputs):
            modes.append((module.training, torch.is_grad_enabled()))

        hooks = [
            agent.target_network.register_forward_pre_hook(record_mode),
            agent.online_network.register_forward_pre_hook(record_mode),
        ]
        try:
            agent.learn([Transition(Features(), 0, 0.1, Features(), False)])
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(modes, [(False, False), (True, True)])
        self.assertFalse(agent.online_network.training)
        self.assertFalse(agent.target_network.training)

    def test_only_online_network_learns_until_scheduled_copy(self):
        agent = make_agent(target_update_every_updates=2)
        before = {name: value.clone() for name, value in agent.target_network.state_dict().items()}
        agent.learn([terminal_transition()])
        self.assertEqual(agent.learning_updates, 1)
        self.assertFalse(torch.equal(agent.online_network.bias, before["bias"]))
        for name, value in agent.target_network.state_dict().items():
            torch.testing.assert_close(value, before[name])
        for parameter in agent.target_network.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        self.assertFalse(agent.target_network.training)
        # Choosing actions does not count toward the target-copy interval.
        agent.select_action(Features())
        self.assertEqual(agent.learning_updates, 1)
        agent.learn([terminal_transition()])
        self.assertEqual(agent.learning_updates, 2)
        for name, value in agent.target_network.state_dict().items():
            torch.testing.assert_close(value, agent.online_network.state_dict()[name])

    def test_target_is_a_separate_copy_and_explicit_sync_includes_buffers(self):
        network = nn.Sequential(nn.BatchNorm1d(2), constant_network())
        agent = DQNAgent(network, device="cpu")
        for online, target in zip(network.parameters(), agent.target_network.parameters()):
            self.assertNotEqual(online.data_ptr(), target.data_ptr())
        with torch.no_grad():
            network[0].running_mean.fill_(3.0)
        agent.update_target_network()
        torch.testing.assert_close(agent.target_network[0].running_mean, network[0].running_mean)

    def test_learning_reduces_terminal_prediction_error_and_restores_eval_mode(self):
        agent = make_agent(optimizer={"learning_rate": 0.01})
        agent.online_network.eval()
        first_loss = agent.learn([terminal_transition()])
        second_loss = agent.learn([terminal_transition()])
        self.assertLess(second_loss, first_loss)
        self.assertFalse(agent.online_network.training)

    def test_invalid_batch_fails_before_learning(self):
        agent = make_agent()
        cases = [
            [],
            [Transition(Features(), 1, 0.0, Features(), False)],
            [Transition(Features(), -1, 0.0, Features(), False)],
            [Transition(Features(), 3, 0.0, Features(), False)],
            [Transition(Features(), True, 0.0, Features(), False)],
            [Transition(Features(), 0, float("nan"), Features(), False)],
            [Transition(Features(), 0, 0.0, Features(action_mask=(False,) * 3), False)],
            [Transition(Features(), 0, 0.0, Features(action_mask=(True,)), False)],
            [terminal_transition(), Transition(Features(action_mask=(True,)), 0, 1.0, Features(), True)],
        ]
        for batch in cases:
            with self.subTest(batch=batch):
                with self.assertRaises(ValueError):
                    agent.learn(batch)
                self.assertEqual(agent.learning_updates, 0)

    def test_invalid_hyperparameters_fail(self):
        cases = [
            dict(optimizer={"learning_rate": 0.0}),
            dict(optimizer={"learning_rate": float("inf")}),
            dict(discount_factor=-0.1), dict(discount_factor=1.1),
            dict(discount_factor=float("nan")),
            dict(target_update_every_updates=0), dict(target_update_every_updates=True),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    make_agent(**kwargs)

    def test_incorrect_output_size_fails_and_restores_network_mode(self):
        agent = DQNAgent(nn.Linear(2, 2), device="cpu")
        with self.assertRaisesRegex(ValueError, "Q-value"):
            agent.select_action(Features())
        self.assertTrue(agent.online_network.training)
        with self.assertRaisesRegex(ValueError, "Q-value"):
            agent.learn([terminal_transition()])
        self.assertEqual(agent.learning_updates, 0)
        self.assertTrue(agent.online_network.training)

    def test_scheduler_advances_only_after_successful_optimizer_updates(self):
        agent = make_agent(
            optimizer={"name": "sgd", "learning_rate": 0.1, "weight_decay": 0.01},
            lr_scheduler={"name": "step", "options": {"step_size": 2, "gamma": 0.5}},
        )
        self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 0.1)
        agent.select_action(Features())
        with self.assertRaises(ValueError):
            agent.learn([])
        self.assertEqual(agent.learning_updates, 0)
        self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 0.1)
        events = []
        original_scheduler_step = agent.lr_scheduler.step

        def scheduler_step(*args, **kwargs):
            events.append("scheduler")
            return original_scheduler_step(*args, **kwargs)

        hook = agent.optimizer.register_step_post_hook(
            lambda optimizer, args, kwargs: events.append("optimizer")
        )
        try:
            with patch.object(agent.lr_scheduler, "step", side_effect=scheduler_step):
                agent.learn([terminal_transition()])
                self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 0.1)
                agent.learn([terminal_transition()])
        finally:
            hook.remove()
        self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 0.05)
        self.assertEqual(events, ["optimizer", "scheduler"] * 2)
        self.assertEqual(agent.learning_updates, 2)

    def test_optimizer_failure_does_not_advance_scheduler(self):
        agent = make_agent(lr_scheduler={"name": "exponential", "options": {"gamma": 0.5}})
        with patch.object(agent.optimizer, "step", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                agent.learn([terminal_transition()])
        self.assertEqual(agent.learning_updates, 0)
        self.assertAlmostEqual(agent.optimizer.param_groups[0]["lr"], 0.001)

    def test_default_optimizer_and_scheduler_preserve_previous_behavior(self):
        agent = make_agent()
        self.assertIsInstance(agent.optimizer, torch.optim.Adam)
        self.assertEqual(agent.optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertIsNone(agent.lr_scheduler)

    def test_nonfinite_loss_aborts_without_optimizer_update(self):
        agent = make_agent()
        with torch.no_grad():
            agent.online_network.bias[0] = torch.inf
        with self.assertRaises(FloatingPointError):
            agent.learn([terminal_transition()])
        self.assertEqual(agent.learning_updates, 0)
        self.assertEqual(agent.optimizer.state_dict()["state"], {})

    def test_tictactoe_adapter_and_replay_buffer_integration(self):
        replies = iter([3, 4])
        environment = TicTacToeAdapter(lambda game: next(replies))
        buffer = ReplayBuffer(10, random.Random(42))
        observation = environment.reset()
        for action in (0, 1, 2):
            next_observation, reward, done = environment.step(action)
            buffer.add(Transition(observation, action, reward, next_observation, done))
            observation = next_observation
        agent = DQNAgent(TicTacToeQNetwork(), device="cpu")
        loss = agent.learn(buffer.sample(3))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        observation = environment.reset()
        self.assertTrue(observation.action_mask[agent.select_action(observation)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_action_selection_and_learning(self):
        agent = DQNAgent(constant_network(), device="cuda")
        self.assertEqual(agent.select_action(Features()), 2)
        self.assertAlmostEqual(agent.learn([terminal_transition()]), 0.32, places=6)
        self.assertEqual(next(agent.target_network.parameters()).device.type, "cuda")
