from __future__ import annotations

import unittest

import torch
from torch import nn

from policies.realman_beaver.modules import StructuredBeaverEncoder


def beaver_inputs(
    *leading_shape: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = torch.full((*leading_shape, 9, 4, 4), 100.0)
    status = torch.full((*leading_shape, 9, 4, 4), 5, dtype=torch.long)
    present = torch.ones(*leading_shape, 9)
    return distance, status, present


class StructuredBeaverEncoderTest(unittest.TestCase):
    def test_variant_input_token_and_output_shapes(self) -> None:
        expected_inputs = {
            "dp_beaver_enc": 33,
            "dp_beaver_near": 49,
            "dp_beaver_near_gate": 49,
        }
        distance, status, present = beaver_inputs(4)
        for variant, input_dim in expected_inputs.items():
            with self.subTest(variant=variant):
                encoder = StructuredBeaverEncoder(variant, output_dim=17)
                output, values = encoder(
                    distance, status, present, return_intermediates=True
                )
                self.assertEqual(values["sensor_input"].shape, (4, 9, input_dim))
                self.assertEqual(values["sensor_tokens"].shape, (4, 9, 32))
                self.assertEqual(output.shape, (4, 17))

    def test_near_field_is_cellwise_and_has_required_mapping(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_near")
        distance, status, present = beaver_inputs(1)
        distance[0, 0, 0] = torch.tensor([0.0, 150.0, 300.0, 600.0])

        _, values = encoder(distance, status, present, return_intermediates=True)

        torch.testing.assert_close(
            values["near"][0, 0, 0], torch.tensor([1.0, 0.5, 0.0, 0.0])
        )

    def test_configured_statuses_distinguish_valid_zero_from_invalid(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_enc", valid_statuses=(7, 11))
        distance = torch.zeros(1, 9, 4, 4)
        status = torch.full((1, 9, 4, 4), 5, dtype=torch.long)
        status[0, 0, 0, 0] = 7
        status[0, 0, 0, 1] = 11
        present = torch.ones(1, 9)

        _, values = encoder(distance, status, present, return_intermediates=True)

        self.assertEqual(values["valid_cell"][0, 0, 0, 0].item(), 1.0)
        self.assertEqual(values["valid_cell"][0, 0, 0, 1].item(), 1.0)
        self.assertEqual(values["valid_cell"][0, 0, 0, 2].item(), 0.0)
        self.assertEqual(values["distance_global"][0, 0, 0, 0].item(), 0.0)
        # In the 33D input, the validity entries follow the 16 distances.
        self.assertEqual(values["sensor_input"][0, 0, 16].item(), 1.0)
        self.assertEqual(values["sensor_input"][0, 0, 18].item(), 0.0)

    def test_near_variant_distinguishes_valid_zero_from_invalid_zero(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_near", valid_statuses=(7,))
        distance = torch.zeros(1, 9, 4, 4)
        status = torch.zeros(1, 9, 4, 4, dtype=torch.long)
        status[0, 0, 0, 0] = 7
        present = torch.ones(1, 9)

        _, values = encoder(distance, status, present, return_intermediates=True)

        self.assertEqual(values["near"][0, 0, 0, 0].item(), 1.0)
        self.assertEqual(values["valid_cell"][0, 0, 0, 0].item(), 1.0)
        self.assertEqual(values["near"][0, 0, 0, 1].item(), 0.0)
        self.assertEqual(values["valid_cell"][0, 0, 0, 1].item(), 0.0)

    def test_global_distance_is_clamped_and_invalid_cells_are_zero(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_enc", distance_max_mm=1000.0)
        distance, status, present = beaver_inputs(1)
        distance[0, 0, 0] = torch.tensor([-10.0, 500.0, 1000.0, 2000.0])
        status[0, 0, 0, 1] = 255

        _, values = encoder(distance, status, present, return_intermediates=True)

        torch.testing.assert_close(
            values["distance_global"][0, 0, 0],
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )

    def test_absent_sensor_is_zero_after_identity_embedding(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_near")
        distance, status, present = beaver_inputs(2)
        present[:, 3] = 0.0

        _, values = encoder(distance, status, present, return_intermediates=True)

        self.assertTrue(torch.equal(values["valid_cell"][:, 3], torch.zeros(2, 4, 4)))
        self.assertTrue(torch.equal(values["sensor_tokens"][:, 3], torch.zeros(2, 32)))

    def test_one_sensor_mlp_is_shared_by_all_physical_sensors(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_enc")
        linear_layers = [
            module
            for module in encoder.sensor_mlp.modules()
            if isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(linear_layers), 3)
        self.assertFalse(
            any(isinstance(module, nn.ModuleList) for module in encoder.modules())
        )

        distance, status, present = beaver_inputs(1)
        with torch.no_grad():
            encoder.sensor_embedding.zero_()
        _, values = encoder(distance, status, present, return_intermediates=True)
        for sensor_id in range(1, 9):
            torch.testing.assert_close(
                values["sensor_tokens"][:, sensor_id],
                values["sensor_tokens"][:, 0],
            )

    def test_only_gated_variant_constructs_sigmoid_sensor_gate(self) -> None:
        plain = StructuredBeaverEncoder("dp_beaver_enc")
        near = StructuredBeaverEncoder("dp_beaver_near")
        self.assertFalse(hasattr(plain, "near_threshold_mm"))
        self.assertFalse(hasattr(plain, "gate_mlp"))
        self.assertFalse(hasattr(near, "gate_mlp"))

        encoder = StructuredBeaverEncoder("dp_beaver_near_gate")
        distance, status, present = beaver_inputs(2, 3)
        present[..., -1] = 0.0
        with torch.no_grad():
            for module in encoder.gate_mlp.modules():
                if isinstance(module, nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()

        output, values = encoder(distance, status, present, return_intermediates=True)

        raw_gate = values["raw_gate"]
        effective_gate = values["effective_gate"]
        self.assertEqual(raw_gate.shape, (2, 3, 9, 1))
        self.assertTrue(torch.all((raw_gate >= 0.0) & (raw_gate <= 1.0)))
        torch.testing.assert_close(raw_gate.sum(dim=-2), torch.full((2, 3, 1), 4.5))
        self.assertFalse(torch.allclose(raw_gate.sum(dim=-2), torch.ones(2, 3, 1)))
        self.assertTrue(torch.equal(effective_gate[..., -1, :], torch.zeros(2, 3, 1)))
        self.assertEqual(output.shape, (2, 3, 64))

    def test_arbitrary_batch_and_time_leading_dimensions_are_preserved(self) -> None:
        encoder = StructuredBeaverEncoder("dp_beaver_near_gate")
        distance, status, present = beaver_inputs(2, 3, 4)

        output, values = encoder(distance, status, present, return_intermediates=True)

        self.assertEqual(output.shape, (2, 3, 4, 64))
        self.assertEqual(values["sensor_input"].shape, (2, 3, 4, 9, 49))
        self.assertEqual(values["sensor_tokens"].shape, (2, 3, 4, 9, 32))

    def test_rejects_unsupported_variants_and_nonpositive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            StructuredBeaverEncoder("dp_beaver")
        with self.assertRaises(ValueError):
            StructuredBeaverEncoder("dp_beaver_enc", output_dim=0)
        with self.assertRaises(ValueError):
            StructuredBeaverEncoder("dp_beaver_near", near_threshold_mm=0.0)


if __name__ == "__main__":
    unittest.main()
