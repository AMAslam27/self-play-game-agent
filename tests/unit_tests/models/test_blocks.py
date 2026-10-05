import unittest

import torch
from torch import nn

from models.blocks import DenseBlock, DropoutBlock, ResidualBlock, build_blocks


class TestBlocks(unittest.TestCase):
    def test_mixed_blocks_preserve_batch_and_support_backpropagation(self):
        blocks = build_blocks(
            9, [16, 16, 8], ["dense", "residual", "dropout"], ["relu", "tanh", "gelu"]
        )
        self.assertIsInstance(blocks[0], DenseBlock)
        self.assertIsInstance(blocks[1], ResidualBlock)
        self.assertIsInstance(blocks[2], DropoutBlock)
        inputs = torch.randn(4, 9, requires_grad=True)
        output = blocks(inputs)
        self.assertEqual(output.shape, (4, 8))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        for parameter in blocks.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_residual_branch_preserves_input_and_gradient(self):
        block = ResidualBlock(8, 8, "relu")
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        inputs = torch.randn(3, 8, requires_grad=True)
        output = block(inputs)
        torch.testing.assert_close(output, inputs)
        output.sum().backward()
        torch.testing.assert_close(inputs.grad, torch.ones_like(inputs))

    def test_dropout_changes_training_outputs_and_is_disabled_in_eval(self):
        block = DropoutBlock(8, 8, "identity", 0.5)
        with torch.no_grad():
            block[0][0].weight.copy_(torch.eye(8))
            block[0][0].bias.zero_()
        inputs = torch.ones(128, 8)
        block.train()
        training_output = block(inputs)
        self.assertTrue((training_output == 0).any())
        self.assertTrue((training_output == 2).any())
        block.eval()
        torch.testing.assert_close(block(inputs), inputs)

    def test_supported_activations_create_separate_modules(self):
        for name, expected_type in [
            ("relu", nn.ReLU), ("tanh", nn.Tanh),
            ("gelu", nn.GELU), ("identity", nn.Identity),
        ]:
            with self.subTest(activation=name):
                blocks = build_blocks(9, [8, 8], ["dense", "dense"], [name, name])
                self.assertIsInstance(blocks[0][1], expected_type)
                self.assertIsNot(blocks[0][1], blocks[1][1])

    def test_invalid_configurations_fail_early(self):
        cases = [
            ({"hidden_sizes": [], "block_types": [], "activations": []}, "At least one"),
            ({"block_types": []}, "equal lengths"),
            ({"activations": []}, "equal lengths"),
            ({"block_types": ["unknown"]}, "Unsupported block type"),
            ({"activations": ["unknown"]}, "Unsupported activation"),
            ({"hidden_sizes": [0]}, "positive integers"),
            ({"hidden_sizes": [-1]}, "positive integers"),
            ({"hidden_sizes": [1.5]}, "positive integers"),
            ({"hidden_sizes": [True]}, "positive integers"),
            ({"input_size": 0}, "positive integers"),
            ({"block_types": ["residual"]}, "matching input and output"),
            ({"dropout_probability": -0.1}, "Dropout probability"),
            ({"dropout_probability": 1.0}, "Dropout probability"),
            ({"dropout_probability": float("nan")}, "Dropout probability"),
        ]
        for overrides, message in cases:
            config = dict(input_size=9, hidden_sizes=[8], block_types=["dense"], activations=["relu"])
            config.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    build_blocks(**config)
