import io
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from models.tictactoe import TicTacToeQNetwork


class TestTicTacToeQNetwork(unittest.TestCase):
    def test_single_board_and_batch_shapes(self):
        model = TicTacToeQNetwork()
        self.assertEqual(model(torch.zeros(9)).shape, (9,))
        self.assertEqual(model(torch.zeros(4, 9)).shape, (4, 9))

    def test_output_head_allows_negative_and_positive_values(self):
        model = TicTacToeQNetwork()
        self.assertIsInstance(model.output, nn.Linear)
        expected = torch.arange(-4, 5, dtype=torch.float32)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.copy_(expected)
        torch.testing.assert_close(model(torch.zeros(9)), expected)

    def test_example_config_builds_a_working_model(self):
        path = Path(__file__).resolve().parents[3] / "training/tictactoe/config.example.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = TicTacToeQNetwork(**config["network"])
        self.assertEqual(model(torch.zeros(2, 9)).shape, (2, 9))

    def test_mixed_model_can_restore_weights_and_predict_identically(self):
        config = dict(
            hidden_sizes=[16, 16, 8],
            block_types=["dense", "residual", "dropout"],
            activations=["relu", "relu", "tanh"],
            dropout_probability=0.2,
        )
        model = TicTacToeQNetwork(**config).eval()
        inputs = torch.randn(4, 9)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = TicTacToeQNetwork(**config).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(restored(inputs), model(inputs))
