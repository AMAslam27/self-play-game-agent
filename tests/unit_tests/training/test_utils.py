import random
import unittest
from dataclasses import FrozenInstanceError

from training.tictactoe.environment import Observation, TicTacToeAdapter
from training.utils import ReplayBuffer, Transition, build_lr_scheduler, build_optimizer, linear_epsilon


def make_transition(index):
    return Transition(
        observation=(index, 0),
        action=index,
        reward=float(index),
        next_observation=(index, 1),
        done=False,
    )


class TestReplayBuffer(unittest.TestCase):
    def test_capacity_must_be_a_positive_integer(self):
        for capacity in (0, -1, 1.5, True, None):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "capacity"):
                    ReplayBuffer(capacity)

    def test_full_buffer_discards_oldest_entries(self):
        buffer = ReplayBuffer(3, random.Random(42))
        self.assertEqual(len(buffer), 0)
        for index in range(5):
            buffer.add(make_transition(index))
        self.assertEqual(len(buffer), 3)
        self.assertEqual({entry.action for entry in buffer.sample(3)}, {2, 3, 4})

    def test_capacity_one_keeps_only_latest_entry(self):
        buffer = ReplayBuffer(1)
        buffer.add(make_transition(0))
        latest = make_transition(1)
        buffer.add(latest)
        self.assertEqual(buffer.sample(1), [latest])

    def test_sampling_does_not_replace_or_remove_entries(self):
        buffer = ReplayBuffer(5, random.Random(42))
        for index in range(5):
            buffer.add(make_transition(index))
        batch = buffer.sample(4)
        self.assertEqual(len(batch), 4)
        self.assertEqual(len({entry.action for entry in batch}), 4)
        self.assertEqual(len(buffer), 5)
        batch.clear()
        self.assertEqual({entry.action for entry in buffer.sample(5)}, set(range(5)))

    def test_invalid_sample_sizes_fail_without_changing_buffer(self):
        buffer = ReplayBuffer(3)
        with self.assertRaisesRegex(ValueError, "Not enough transitions"):
            buffer.sample(1)
        buffer.add(make_transition(0))
        for size in (0, -1, 1.5, True, None, 2):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    buffer.sample(size)
                self.assertEqual(len(buffer), 1)

    def test_seeded_generators_reproduce_sample_sequences(self):
        first = ReplayBuffer(10, random.Random(42))
        second = ReplayBuffer(10, random.Random(42))
        for index in range(10):
            first.add(make_transition(index))
            second.add(make_transition(index))
        for _ in range(5):
            self.assertEqual(first.sample(4), second.sample(4))

    def test_default_sampling_does_not_advance_global_random_state(self):
        state = random.getstate()
        buffer = ReplayBuffer(3)
        for index in range(3):
            buffer.add(make_transition(index))
        buffer.sample(2)
        self.assertEqual(random.getstate(), state)

    def test_transition_fields_cannot_be_reassigned(self):
        transition = make_transition(0)
        with self.assertRaises(FrozenInstanceError):
            transition.reward = 1.0

    def test_adapter_transitions_preserve_snapshots_and_terminal_masks(self):
        replies = iter([3, 4])
        adapter = TicTacToeAdapter(lambda game: next(replies))
        buffer: ReplayBuffer[Observation] = ReplayBuffer(10, random.Random(42))
        observation = adapter.reset()
        for action in (0, 1, 2):
            next_observation, reward, done = adapter.step(action)
            buffer.add(Transition(observation, action, reward, next_observation, done))
            observation = next_observation
        adapter.reset()

        entries = {entry.action: entry for entry in buffer.sample(3)}
        self.assertEqual(entries[0].observation.board, (0,) * 9)
        self.assertEqual(entries[0].next_observation.board, (1, 0, 0, -1, 0, 0, 0, 0, 0))
        self.assertEqual(entries[2].reward, 1.0)
        self.assertTrue(entries[2].done)
        self.assertEqual(entries[2].next_observation.action_mask, (False,) * 9)


class TestOptimizationConfigValidation(unittest.TestCase):
    def test_invalid_optimizer_numbers_fail_before_construction(self):
        for config in [
            {"learning_rate": 0}, {"learning_rate": float("nan")},
            {"learning_rate": True}, {"learning_rate": "0.01"},
            {"weight_decay": -0.1}, {"weight_decay": float("inf")},
        ]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    build_optimizer([], config)

    def test_invalid_optimizer_mapping_and_unknown_fields_fail(self):
        for config in ("adam", {"weight_deacy": 0.1}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    build_optimizer([], config)

    def test_optimizer_options_cannot_override_explicit_settings(self):
        for options in (None, [], {"lr": 0.1}, {"weight_decay": 0.1}, {"params": []}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    build_optimizer([], {"options": options})

    def test_disabled_scheduler_requires_no_optimizer_operations(self):
        self.assertIsNone(build_lr_scheduler(None))
        self.assertIsNone(build_lr_scheduler(None, {"name": "none", "options": {}}))

    def test_invalid_scheduler_mapping_or_disabled_options_fail(self):
        for config in ("none", {"extra": 1}, {"options": []}, {"options": {"gamma": 0.5}}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    build_lr_scheduler(None, config)


class TestReplayStateAndEpsilon(unittest.TestCase):
    def test_replay_restores_future_sampling_and_fifo_order(self):
        original = ReplayBuffer(3, random.Random(42))
        for index in range(5):
            original.add(make_transition(index))
        original.sample(2)
        state = original.state_dict()
        restored = ReplayBuffer(3)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        for _ in range(3):
            self.assertEqual(original.sample(2), restored.sample(2))
        original.add(make_transition(5))
        restored.add(make_transition(5))
        self.assertEqual(original.state_dict()["transitions"], restored.state_dict()["transitions"])

    def test_mismatched_capacity_is_rejected(self):
        with self.assertRaises(ValueError):
            ReplayBuffer(2).load_state_dict(ReplayBuffer(3).state_dict())

    def test_epsilon_boundaries_and_clamping(self):
        self.assertEqual(linear_epsilon(0, 1.0, 0.1, 100), 1.0)
        self.assertAlmostEqual(linear_epsilon(50, 1.0, 0.1, 100), 0.55)
        self.assertAlmostEqual(linear_epsilon(100, 1.0, 0.1, 100), 0.1)
        self.assertAlmostEqual(linear_epsilon(200, 1.0, 0.1, 100), 0.1)
        for arguments in ((-1, 1, 0.1, 100), (1, 0.1, 1, 100), (1, 1, 0, 0)):
            with self.assertRaises(ValueError):
                linear_epsilon(*arguments)


if __name__ == "__main__":
    unittest.main()
