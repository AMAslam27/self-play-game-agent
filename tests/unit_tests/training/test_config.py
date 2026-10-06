import copy
import tempfile
import unittest
from pathlib import Path

from training.config import PROJECT_ROOT, resolve_config


class TestTrainingConfig(unittest.TestCase):
    def test_defaults_and_overrides_are_resolved_without_mutation(self):
        original = {"training": {"optimizer": {"name": "sgd", "options": {"momentum": 0.9}}}}
        supplied = copy.deepcopy(original)
        config = resolve_config(supplied, Path("training/tictactoe/runs"), episodes=8, device="cpu")
        self.assertEqual(config["training"]["episodes"], 8)
        self.assertEqual(config["experiment"]["device"], "cpu")
        self.assertEqual(config["training"]["optimizer"]["weight_decay"], 0.0)
        self.assertEqual(config["training"]["optimizer"]["options"], {"momentum": 0.9})
        self.assertEqual(supplied, original)
        self.assertEqual(config["artifacts"]["output_root"], str(PROJECT_ROOT / "training/tictactoe/runs"))

    def test_absolute_output_root_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            config = resolve_config({"artifacts": {"output_root": directory}}, Path("unused"))
            self.assertEqual(Path(config["artifacts"]["output_root"]), Path(directory).resolve())

    def test_warmup_can_exceed_replay_capacity(self):
        config = resolve_config({"training": {"replay_capacity": 64, "learning_starts": 1000}}, Path("runs"))
        self.assertEqual(config["training"]["learning_starts"], 1000)

    def test_step_scheduler_default_gamma_is_persisted(self):
        config = resolve_config({"training": {"lr_scheduler": {"name": "step", "options": {"step_size": 5}}}}, Path("runs"))
        self.assertEqual(config["training"]["lr_scheduler"]["options"]["gamma"], 0.1)

    def test_invalid_settings_are_rejected(self):
        cases = [
            {"unknown": 1}, {"schema_version": 9}, {"experiment": None},
            {"experiment": {"name": ""}}, {"experiment": {"seed": -1}},
            {"experiment": {"seed": True}}, {"experiment": {"device": "magic"}},
            {"experiment": {"deterministic": "yes"}},
            {"training": {"batch_size": 0}}, {"training": {"replay_capacity": 3}},
            {"training": {"learning_starts": 0}}, {"training": {"loss": "unknown"}},
            {"training": {"optimizer": {"weight_decay": -0.1}}},
            {"training": {"lr_scheduler": {"name": "step", "options": {"step_size": 0}}}},
            {"training": {"lr_scheduler": {"name": "exponential"}}},
            {"network": {"hidden_sizes": [4]}}, {"network": {"activations": ["magic", "relu"]}},
            {"exploration": {"final_epsilon": 1.1}}, {"exploration": {"schedule": "magic"}},
            {"evaluation": {"seats": ["x", "x"]}}, {"evaluation": {"epsilon": 0.1}},
            {"evaluation": {"opponents": ["human"]}},
            {"artifacts": {"plot_every_episodes": 0}}, {"artifacts": {"output_root": ""}},
        ]
        for supplied in cases:
            with self.subTest(supplied=supplied):
                with self.assertRaises(ValueError):
                    resolve_config(supplied, Path("runs"))
