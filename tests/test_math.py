"""独立小张量期望值与临时 32KB 曲线文件；不读取安装目录中的模型权重。"""

from contextlib import contextmanager
import importlib
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .support import isolated_plugin


class ProjectionMappingTests(unittest.TestCase):
    def test_qkv_offsets_produce_independent_fused_projection(self):
        with isolated_plugin("adapters") as env:
            torch = env.torch
            a = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], device="cpu")
            b = torch.tensor([[2.0, 0.0], [1.0, 3.0]], device="cpu")
            a_before, b_before = a.clone(), b.clone()
            cases = {
                "q": (0, [[2.0, 4.0], [-2.0, 5.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
                "k": (2, [[0.0, 0.0], [0.0, 0.0], [2.0, 4.0], [-2.0, 5.0], [0.0, 0.0], [0.0, 0.0]]),
                "v": (4, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 4.0], [-2.0, 5.0]]),
            }
            for part, (start, expected) in cases.items():
                with self.subTest(part=part):
                    mappings = env.adapters.mapped_parts(f"transformer.transformer_blocks.0.attn.to_{part}", a, b)
                    self.assertEqual(len(mappings), 1)
                    name, offset, down, up = mappings[0]
                    self.assertEqual(name, "blocks.0.attn.qkv_proj")
                    self.assertEqual(offset, (0, start, 2))
                    fused = torch.zeros((6, 2), device="cpu")
                    fused.narrow(*offset).add_(up @ down)
                    torch.testing.assert_close(fused, torch.tensor(expected))
            torch.testing.assert_close(a, a_before)
            torch.testing.assert_close(b, b_before)

    def test_mlp_gate_and_value_rows_are_swapped_numerically(self):
        with isolated_plugin("adapters") as env:
            torch = env.torch
            a = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
            b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]])
            before = b.clone()
            mappings = env.adapters.mapped_parts("transformer.transformer_blocks.2.ff.net.0.proj", a, b)
            self.assertEqual(len(mappings), 2)
            fused = torch.zeros((4, 2))
            for name, offset, down, up in mappings:
                self.assertEqual(name, "blocks.2.mlp.fc1")
                fused.narrow(*offset).add_(up @ down)
            torch.testing.assert_close(fused, torch.tensor([[2.0, 4.0], [-3.0, 3.0], [1.0, 2.0], [-1.0, 1.0]]))
            torch.testing.assert_close(b, before)

    def test_mlp_odd_output_rows_are_rejected(self):
        with isolated_plugin("adapters") as env:
            with self.assertRaisesRegex(ValueError, "偶数"):
                env.adapters.mapped_parts("blocks.0.ff.net.0.proj", env.torch.ones((1, 2)), env.torch.ones((3, 1)))

    def test_output_projections_and_token_refiner_names(self):
        with isolated_plugin("adapters") as env:
            a, b = env.torch.ones((1, 2)), env.torch.ones((2, 1))
            for source, expected in [
                ("transformer.transformer_blocks.1.attn.to_out.0", "blocks.1.attn.out_proj"),
                ("transformer.transformer_blocks.1.ff.net.2", "blocks.1.mlp.fc2"),
                ("transformer.token_refiner.refiner_blocks.3.attn.to_q", "token_refiner.blocks.3.attn.qkv_proj"),
                ("transformer.norm_out.linear", "final_layer.adaln_proj.linear"),
            ]:
                with self.subTest(source=source):
                    self.assertEqual(env.adapters.mapped_parts(source, a, b)[0][0], expected)


class CurveAndScheduleTests(unittest.TestCase):
    def test_curve_endpoints_clamping_and_linear_interpolation(self):
        with isolated_plugin("patches") as env:
            torch = env.torch
            table = torch.tensor([[0.0, 10.0], [4.0, 6.0], [12.0, 2.0]])
            times = torch.tensor([-1.0, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0], requires_grad=True)
            before = times.detach().clone()
            expected = torch.tensor([[0.0, 10.0], [0.0, 10.0], [1.0, 9.0], [2.0, 8.0],
                                     [4.0, 6.0], [8.0, 4.0], [12.0, 2.0], [12.0, 2.0]])
            result = env.patches.curve_coordinates(table, times)
            torch.testing.assert_close(result, expected)
            torch.testing.assert_close(times.detach(), before)
            self.assertFalse(result.requires_grad)
            self.assertEqual(result.device.type, "cpu")

    def test_reference_two_and_three_step_sigmas(self):
        with isolated_plugin("schedules") as env:
            two = env.schedules.schedule_values("reference_2", steps=99, start_sigma=0.1, shift=2.0)
            three = env.schedules.schedule_values("reference_3", steps=99, start_sigma=0.1, shift=2.0)
            for actual, expected in zip(two, [0.7, 42 / 85, 0.0]):
                self.assertAlmostEqual(actual, expected, places=12)
            for actual, expected in zip(three, [291 / 320, 6 / 7, 2 / 3, 0.0]):
                self.assertAlmostEqual(actual, expected, places=12)
            self.assertEqual(len(two), 3)
            self.assertEqual(len(three), 4)
            env.schedules.validate_values(two)
            env.schedules.validate_values(three)

    def test_shifted_and_custom_sigma_values(self):
        with isolated_plugin("schedules") as env:
            self.assertEqual(env.schedules.schedule_values("shifted", steps=2, start_sigma=1.0, shift=1.0), [1.0, 0.5, 0.0])
            self.assertEqual(env.schedules.schedule_values("custom", custom="0.9; 0.5, 0"), [0.9, 0.5, 0.0])

    def test_invalid_sigma_sequences_and_parameters(self):
        with isolated_plugin("schedules") as env:
            for values in ([0.7], [0.7, 0.2], [0.7, 0.7, 0.0], [0.0, 0.7], [1.1, 0.0], [-0.1, 0.0], [float("nan"), 0.0], [float("inf"), 0.0]):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    env.schedules.validate_values(values)
            for kwargs in ({"steps": 0}, {"shift": 0}, {"start_sigma": 0}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    env.schedules.schedule_values("shifted", **kwargs)

    def test_latent_frame_count_37_corresponds_to_124_frames(self):
        with isolated_plugin("sampling") as env:
            for tokens, frames in [(1, 1), (2, 5), (4, 13), (5, 17), (6, 18), (35, 119), (36, 120), (37, 124)]:
                with self.subTest(tokens=tokens):
                    self.assertEqual(env.sampling.video_frames(tokens), frames)


class SmallWeightFileTests(unittest.TestCase):
    def test_safetensors_reads_only_named_curve_tensor(self):
        with isolated_plugin("weights") as env, tempfile.TemporaryDirectory(prefix="ck_h3_curve_") as directory:
            torch = env.torch
            save_file = importlib.import_module("safetensors.torch").save_file
            expected = torch.arange(8200, dtype=torch.float32).reshape(1025, 8)
            real_open = env.weights.safe_open
            reads = []

            @contextmanager
            def tracked_open(*args, **kwargs):
                self.assertEqual(kwargs, {"framework": "pt", "device": "cpu"})
                with real_open(*args, **kwargs) as reader:
                    def get_tensor(key):
                        reads.append(key)
                        self.assertNotEqual(key, "unrelated.weight")
                        return reader.get_tensor(key)
                    yield SimpleNamespace(keys=reader.keys, get_tensor=get_tensor)

            for index, key in enumerate(("adaln_t_table", "diffusion_model.adaln_t_table", "model.diffusion_model.adaln_t_table")):
                with self.subTest(key=key):
                    path = Path(directory) / f"curve_{index}.safetensors"
                    save_file({key: expected, "unrelated.weight": torch.ones((2, 2))}, str(path))
                    with patch.object(env.weights, "safe_open", tracked_open):
                        actual = env.weights.read_curve_file(str(path))
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(reads[-1:], [key])
                    self.assertEqual(actual.device.type, "cpu")
                    del actual
            self.assertEqual(len(reads), 3)

    def test_bin_reads_independent_fp32_values_and_owns_storage(self):
        with isolated_plugin("weights") as env, tempfile.TemporaryDirectory(prefix="ck_h3_bin_") as directory:
            path = Path(directory) / "adaln_t_table.bin"
            path.write_bytes(struct.pack("<8200f", *range(8200)))
            actual = env.weights.read_curve_file(str(path))
            env.torch.testing.assert_close(actual, env.torch.arange(8200, dtype=env.torch.float32).reshape(1025, 8))
            path.write_bytes(b"\x00" * 32800)
            self.assertEqual(actual[0, 1].item(), 1.0)
            self.assertEqual(actual[-1, -1].item(), 8199.0)
            self.assertEqual(actual.device.type, "cpu")

    def test_invalid_bin_size_and_safetensors_shape_or_key(self):
        with isolated_plugin("weights") as env, tempfile.TemporaryDirectory(prefix="ck_h3_bad_") as directory:
            path = Path(directory) / "invalid.bin"
            path.write_bytes(b"\x00" * 4)
            with self.assertRaisesRegex(ValueError, "32800"):
                env.weights.read_curve_file(str(path))
            save_file = importlib.import_module("safetensors.torch").save_file
            path = Path(directory) / "invalid.safetensors"
            save_file({"adaln_t_table": env.torch.zeros((2, 8))}, str(path))
            with self.assertRaisesRegex(ValueError, "形状"):
                env.weights.read_curve_file(str(path))
            path = Path(directory) / "missing_key.safetensors"
            save_file({"unrelated.weight": env.torch.zeros((2, 2))}, str(path))
            with self.assertRaisesRegex(KeyError, "adaln_t_table"):
                env.weights.read_curve_file(str(path))

    def test_absolute_curve_path_does_not_scan_model_directories(self):
        with isolated_plugin("weights") as env, tempfile.TemporaryDirectory(prefix="ck_h3_path_") as directory:
            path = Path(directory) / "curve.bin"
            path.write_bytes(b"\x00" * 32800)
            resolved = env.weights.resolve_file(str(path), ("diffusion_models", "loras"))
            self.assertEqual(Path(resolved), path.resolve())
