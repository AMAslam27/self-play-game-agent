import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from training.checkpoints import capture_rng_state, load_checkpoint, preserve_rng_state
from training.config import load_config, resolve_config
from training.metrics import read_metrics
from training.plots import plot_training_metrics
from training.tictactoe.train import DEFAULT_CONFIG_PATH, DEFAULT_RUNS_DIR, run_training


def short_config(root, episodes=6):
    return resolve_config({
        "experiment": {"seed": 7, "device": "cpu"},
        "network": {"hidden_sizes": [8, 8], "block_types": ["dense", "dropout"]},
        "training": {
            "episodes": episodes, "batch_size": 2, "replay_capacity": 5,
            "learning_starts": 2, "target_update_every_updates": 2,
            "optimizer": {"name": "adamw", "learning_rate": 0.01, "weight_decay": 0.01},
            "lr_scheduler": {"name": "step", "options": {"step_size": 3, "gamma": 0.9}},
        },
        "exploration": {"decay_steps": 10},
        "evaluation": {"every_episodes": 2, "opponents": ["random"], "games_per_opponent_per_seat": 2},
        "artifacts": {
            "output_root": str(root), "metrics_every_episodes": 1,
            "checkpoint_every_episodes": 2, "plot_every_episodes": 1000,
            "save_code_version": False,
        },
    }, DEFAULT_RUNS_DIR)


class TestTrainingPipeline(unittest.TestCase):
    def assert_same_training(self, first, second):
        for key in ("online_network", "target_network"):
            for name, value in first["agent"][key].items():
                torch.testing.assert_close(value, second["agent"][key][name], rtol=0, atol=0)
        self.assertEqual(first["agent"]["learning_updates"], second["agent"]["learning_updates"])
        self.assertEqual(first["agent"]["rng_state"], second["agent"]["rng_state"])
        self.assertEqual(first["agent"]["lr_scheduler"], second["agent"]["lr_scheduler"])
        self.assertEqual(first["replay"], second["replay"])
        self.assertEqual(first["opponent_rng"], second["opponent_rng"])
        torch.testing.assert_close(first["rng"]["torch"], second["rng"]["torch"], rtol=0, atol=0)

    def test_short_run_creates_metrics_evaluations_checkpoints_and_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = short_config(root, episodes=4)
            run = run_training(config)
            metadata = json.loads((run / "run.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["progress"]["episodes"], 4)
            self.assertGreater(metadata["learning_updates"], 0)
            self.assertGreater(metadata["progress"]["training_seconds"], 0)
            self.assertGreater(metadata["progress"]["evaluation_seconds"], 0)
            self.assertGreater(metadata["progress"]["wall_seconds"], metadata["progress"]["training_seconds"])
            self.assertEqual(run.parent, root)
            self.assertTrue((run / "config.yaml").exists())
            self.assertTrue((run / "games.sqlite3").exists())
            self.assertEqual([row["seat"] for row in read_metrics(run, "episodes")], ["x", "o", "x", "o"])
            updates = read_metrics(run, "updates")
            self.assertLess(float(updates[-1]["lr_next"]), float(updates[0]["lr_used"]))
            decisions = read_metrics(run, "decisions")
            self.assertEqual(float(decisions[0]["epsilon"]), 1.0)
            self.assertLess(float(decisions[-1]["epsilon"]), 1.0)
            evaluation = read_metrics(run, "evaluation")
            self.assertEqual({int(row["episode"]) for row in evaluation}, {0, 2, 4})
            self.assertEqual(len(evaluation), 6)
            for row in evaluation:
                self.assertEqual(int(row["wins"]) + int(row["draws"]) + int(row["losses"]), 2)
            saved = load_checkpoint(run / "checkpoints/latest.pt")
            self.assertEqual(saved["progress"]["episodes"], 4)
            self.assertEqual(len(saved["replay"]["transitions"]), 5)
            for path in (run / "plots").glob("*.png"):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(len(list((run / "plots").glob("*.png"))), 7)

    def test_interruption_and_resume_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.tictactoe.train.plot_training_metrics"):
            root = Path(directory)
            full = run_training(short_config(root / "full"))
            interrupted_root = root / "interrupted"

            def stop():
                runs = list(interrupted_root.glob("*"))
                return bool(runs) and len(read_metrics(runs[0], "episodes")) >= 3

            partial = run_training(short_config(interrupted_root), should_stop=stop)
            self.assertEqual(json.loads((partial / "run.json").read_text())["status"], "interrupted")
            resume = partial / "checkpoints/latest.pt"
            checkpoint = load_checkpoint(resume)
            self.assertEqual(checkpoint["progress"]["episodes"], 3)
            run_training(checkpoint["config"], resume=resume)
            final = load_checkpoint(partial / "checkpoints/latest.pt")
            self.assert_same_training(load_checkpoint(full / "checkpoints/latest.pt"), final)
            self.assertEqual([int(row["episode"]) for row in read_metrics(partial, "episodes")], list(range(1, 7)))
            self.assertEqual(len(json.loads((partial / "run.json").read_text())["sessions"]), 2)

    def test_evaluation_frequency_does_not_change_training(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.tictactoe.train.plot_training_metrics"):
            root = Path(directory)
            first_config = short_config(root / "first")
            second_config = short_config(root / "second")
            second_config["evaluation"]["every_episodes"] = 3
            first = run_training(first_config)
            second = run_training(second_config)
            self.assert_same_training(load_checkpoint(first / "checkpoints/latest.pt"), load_checkpoint(second / "checkpoints/latest.pt"))

    def test_resume_older_checkpoint_archives_history_and_avoids_duplicate_metrics(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.tictactoe.train.plot_training_metrics"):
            run = run_training(short_config(Path(directory)))
            expected = load_checkpoint(run / "checkpoints/latest.pt")
            resume = next((run / "checkpoints").glob("episode-00000002_*.pt"))
            config = load_checkpoint(resume)["config"]
            run_training(config, resume=resume)
            self.assert_same_training(expected, load_checkpoint(run / "checkpoints/latest.pt"))
            self.assertEqual([int(row["episode"]) for row in read_metrics(run, "episodes")], list(range(1, 7)))
            self.assertTrue(list((run / "recoveries").glob("*/episodes.csv")))

    def test_resume_can_extend_target_but_rejects_other_hyperparameter_changes(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.tictactoe.train.plot_training_metrics"):
            run = run_training(short_config(Path(directory), episodes=2))
            resume = run / "checkpoints/latest.pt"
            config = load_checkpoint(resume)["config"]
            config["training"]["episodes"] = 3
            run_training(config, resume=resume)
            self.assertEqual(len(read_metrics(run, "episodes")), 3)
            bad_config = copy.deepcopy(config)
            bad_config["training"]["discount_factor"] = 0.5
            with self.assertRaises(ValueError):
                run_training(bad_config, resume=run / "checkpoints/latest.pt")

    def test_warmup_logs_no_fabricated_loss_and_still_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            config = short_config(Path(directory), episodes=1)
            config["training"]["learning_starts"] = 1000
            run = run_training(config)
            self.assertEqual(read_metrics(run, "updates"), [])
            self.assertTrue((run / "plots/loss.png").exists())

    def test_failure_preserves_initial_checkpoint_and_marks_run_failed(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.tictactoe.train.plot_training_metrics"):
            root = Path(directory)
            with patch("training.tictactoe.train.DQNAgent.learn", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    run_training(short_config(root))
            run = next(root.iterdir())
            self.assertEqual(json.loads((run / "run.json").read_text())["status"], "failed")
            self.assertEqual(load_checkpoint(run / "checkpoints/latest.pt")["progress"]["episodes"], 0)

    def test_example_loads_and_uses_the_requested_runs_directory(self):
        config = load_config(DEFAULT_CONFIG_PATH, DEFAULT_RUNS_DIR, episodes=2, device="cpu")
        self.assertEqual(Path(config["artifacts"]["output_root"]), DEFAULT_RUNS_DIR)

    def test_rng_preservation_restores_python_numpy_and_torch(self):
        state = capture_rng_state()
        with preserve_rng_state():
            random.random()
            np.random.random()
            torch.rand(3)
        restored = capture_rng_state()
        self.assertEqual(state["python"], restored["python"])
        self.assertEqual(state["numpy"], restored["numpy"])
        torch.testing.assert_close(state["torch"], restored["torch"], rtol=0, atol=0)

    def test_plots_can_be_regenerated_from_raw_csvs(self):
        with tempfile.TemporaryDirectory() as directory, patch("training.tictactoe.train.plot_training_metrics"):
            run = run_training(short_config(Path(directory), episodes=2))
            self.assertEqual(len(plot_training_metrics(run, window_episodes=2)), 7)
