"""导入隔离用独立进程；hook 用真实 CPU Linear，采样仅验证受控协议的清理。"""

import json
import subprocess
import sys
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .support import ROOT, PatcherDouble, isolated_plugin


IMPORT_PROBE = textwrap.dedent(r'''
    import builtins
    import importlib.abc
    import importlib.util
    import json
    import logging
    from pathlib import Path
    import socket
    import sys

    blocked = {"torch", "comfy", "folder_paths", "nodes", "safetensors"}
    attempts = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            attempts.append(name)
            raise AssertionError("禁止访问宿主依赖：" + name)
        return original_import(name, *args, **kwargs)

    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked:
                attempts.append(fullname)
                raise AssertionError("禁止动态导入宿主：" + fullname)

    def audit(event, args):
        forbidden = {"socket.connect", "socket.getaddrinfo", "subprocess.Popen",
                     "os.system", "os.mkdir", "os.remove", "os.rename", "urllib.Request"}
        if event in forbidden:
            attempts.append(event)
            raise AssertionError("导入阶段不应下载、启动进程或修改文件：" + event)
        if event == "open":
            mode, flags = args[1:3]
            if (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
                isinstance(flags, int) and flags & (1 | 2 | 64 | 512 | 1024)
            ):
                attempts.append("文件写入")
                raise AssertionError("导入阶段不应写文件")

    sys.dont_write_bytecode = True
    builtins.__import__ = guarded_import
    guard = Guard()
    sys.meta_path.insert(0, guard)
    sys.addaudithook(audit)
    before_builtins = builtins.__dict__.copy()
    before_path = list(sys.path)
    before_meta = list(sys.meta_path)
    before_handlers = list(logging.getLogger().handlers)
    before_socket = (socket.socket, socket.create_connection, socket.getaddrinfo)
    before_hooks = (sys.excepthook, sys.displayhook, sys.unraisablehook)
    root = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location(
        "_ck_import_probe", root / "__init__.py", submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    if sys.argv[2] == "schema":
        for node in package.NODE_CLASS_MAPPINGS.values():
            schema = node.INPUT_TYPES()
            assert isinstance(schema, dict) and "required" in schema
    assert attempts == [], attempts
    assert not any(name.split(".")[0] in blocked for name in sys.modules)
    assert builtins.__dict__ == before_builtins, "builtins 被全局替换"
    assert sys.path == before_path and sys.meta_path == before_meta, "导入路径被修改"
    assert logging.getLogger().handlers == before_handlers, "全局日志处理器被修改"
    assert (socket.socket, socket.create_connection, socket.getaddrinfo) == before_socket
    assert (sys.excepthook, sys.displayhook, sys.unraisablehook) == before_hooks
    assert not any(name.startswith("_ck_import_probe.open_h3") for name in sys.modules)
    print(json.dumps(sorted(package.NODE_CLASS_MAPPINGS)))
''')


class ImportIsolationTests(unittest.TestCase):
    def _probe(self, mode):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-X", "utf8", "-c", IMPORT_PROBE, str(ROOT), mode],
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout), sorted([
            "CKOpenH3ApplyRegenerate", "CKOpenH3RefineContext", "CKOpenH3SigmaSchedule",
            "CKOpenH3RefineSampler", "CKOpenH3VideoLatent", "CKOpenH3EncodeSilence",
            "CKOpenH3PrepareVideo", "CKOpenH3Decode", "CKOpenH3UpscalerLoader", "CKOpenH3LatentUpscale",
        ]))

    def test_plugin_import_has_no_host_or_global_side_effects(self):
        self._probe("import")

    def test_all_input_types_avoid_host_access_and_registration(self):
        self._probe("schema")

    def test_stub_modules_restore_after_success_and_exception(self):
        def host_snapshot():
            return {name: module for name, module in tuple(sys.modules.items())
                    if name == "comfy" or name.startswith("comfy.") or name == "folder_paths"}

        for fail in (False, True):
            with self.subTest(fail=fail):
                before = host_snapshot()
                private_before = {name for name in sys.modules if name.startswith("_ck_open_h3_test_")}
                try:
                    with isolated_plugin("patches") as env:
                        self.assertIs(sys.modules["comfy"], env.modules["comfy"])
                        if fail:
                            raise RuntimeError("验证 finally")
                except RuntimeError as exc:
                    if not fail or str(exc) != "验证 finally":
                        raise
                self.assertEqual(host_snapshot(), before)
                self.assertEqual({name for name in sys.modules if name.startswith("_ck_open_h3_test_")}, private_before)


class SharpHookTests(unittest.TestCase):
    def _fixture(self, env):
        torch = env.torch
        linear = torch.nn.Linear(2, 2, device="cpu")
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[2.0, -1.0], [1.0, 3.0]]))
            linear.bias.copy_(torch.tensor([1.0, -2.0]))
        block = SimpleNamespace(adaln_proj=SimpleNamespace(linear=linear))
        args = {
            "transformer_options": {env.patches.COORDS: torch.tensor([[1.0, 2.0], [0.0, -1.0]]),
                                    env.patches.CORE: SimpleNamespace(blocks=[block])},
            "t_emb": torch.tensor([[1.0, 2.0], [3.0, -1.0]]),
        }
        curve = env.patches.SharpBlockPatch(
            0, torch.tensor([[1.0, 2.0], [-1.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [1.0, -1.0]]), 0.5,
        )
        return linear, args, curve

    def test_real_linear_hook_adds_known_delta_and_restores_existing_hook(self):
        with isolated_plugin("patches") as env:
            linear, args, curve = self._fixture(env)
            seen = []
            original = linear.register_forward_hook(lambda module, inputs, output: seen.append(output.detach().clone()))
            try:
                hooks = dict(linear._forward_hooks)
                result = curve(args, {"original_block": lambda payload: linear(payload["t_emb"])})
                env.torch.testing.assert_close(result, env.torch.tensor([[6.0, 7.0], [6.0, -2.5]]))
                self.assertEqual(dict(linear._forward_hooks), hooks)
                env.torch.testing.assert_close(linear(args["t_emb"]), env.torch.tensor([[1.0, 5.0], [8.0, -2.0]]))
                self.assertEqual(len(seen), 2)
            finally:
                original.remove()

    def test_hook_is_removed_when_original_or_previous_patch_raises(self):
        for previous in (False, True):
            with self.subTest(previous=previous), isolated_plugin("patches") as env:
                linear, args, curve = self._fixture(env)
                hooks = dict(linear._forward_hooks)

                def fail(payload, extra=None):
                    self.assertEqual(len(linear._forward_hooks), len(hooks) + 1)
                    linear(payload["t_emb"])
                    raise RuntimeError("预期的 block 异常")

                if previous:
                    curve.previous = fail
                with self.assertRaisesRegex(RuntimeError, "预期的 block 异常"):
                    curve(args, {"original_block": fail})
                self.assertEqual(dict(linear._forward_hooks), hooks)
                env.torch.testing.assert_close(linear(args["t_emb"]), env.torch.tensor([[1.0, 5.0], [8.0, -2.0]]))

    def test_previous_patch_receives_arguments_and_preserves_hook_cleanup(self):
        with isolated_plugin("patches") as env:
            linear, args, curve = self._fixture(env)
            previous = Mock(side_effect=lambda payload, extra: linear(payload["t_emb"]))
            curve.previous = previous
            extra = {"original_block": Mock(side_effect=AssertionError("不应跳过已有补丁"))}
            result = curve(args, extra)
            self.assertIs(previous.call_args.args[0], args)
            self.assertIs(previous.call_args.args[1], extra)
            self.assertFalse(linear._forward_hooks)
            env.torch.testing.assert_close(result, env.torch.tensor([[6.0, 7.0], [6.0, -2.5]]))

    def test_time_row_mismatch_does_not_install_hook(self):
        with isolated_plugin("patches") as env:
            linear, args, curve = self._fixture(env)
            args["t_emb"] = args["t_emb"][:1]
            with self.assertRaisesRegex(ValueError, "时间行"):
                curve(args, {"original_block": Mock()})
            self.assertFalse(linear._forward_hooks)

    def test_sharp_uses_the_current_forward_model_copy(self):
        with isolated_plugin("patches") as env:
            original, args, curve = self._fixture(env)
            actual = env.torch.nn.Linear(2, 2)
            actual.load_state_dict(original.state_dict())
            args["transformer_options"][env.patches.CORE] = SimpleNamespace(
                blocks=[SimpleNamespace(adaln_proj=SimpleNamespace(linear=actual))])
            result = curve(args, {"original_block": lambda payload: actual(payload["t_emb"])})
            env.torch.testing.assert_close(result, env.torch.tensor([[6.0, 7.0], [6.0, -2.5]]))
            self.assertFalse(original._forward_hooks)
            self.assertFalse(actual._forward_hooks)
            env.torch.testing.assert_close(original(args["t_emb"]), env.torch.tensor([[1.0, 5.0], [8.0, -2.0]]))



class ForwardIsolationTests(unittest.TestCase):
    def test_forward_copies_options_payload_and_preserves_input_audio(self):
        with isolated_plugin("patches") as env:
            torch, patches = env.torch, env.patches
            video = torch.zeros((1, 24, 1, 2, 2), device="cpu")
            audio = torch.full((1, 32, 2, 4), 7.0, device="cpu")
            fixed = torch.full((1, 32, 2, 3), 2.0, device="cpu")
            audio_before, fixed_before = audio.clone(), fixed.clone()
            x = [video, audio]
            marker = object()
            options = {"minimax_h3_sigma_shift_video": 6.0, "keep": marker}
            payload = {"audio_scale": 0.25, "keep": marker}
            calls = []

            def execute(actual_x, timestep, context, actual_options, **kwargs):
                calls.append(True)
                self.assertIsNot(actual_options, options)
                self.assertIsNot(kwargs["minimax_payload"], payload)
                self.assertIsNot(actual_x, x)
                self.assertIs(actual_x[0], video)
                torch.testing.assert_close(actual_x[1], fixed_before)
                self.assertEqual(actual_options["minimax_h3_sigma_shift_audio"], 6.0)
                self.assertEqual(kwargs["minimax_payload"]["audio_scale"], 1.0)
                torch.testing.assert_close(kwargs["audio_denoise_mask"], torch.zeros((1, 1, 2, 3)))
                actual_options["new"] = True
                kwargs["minimax_payload"]["new"] = True
                return [video + 1, actual_x[1] + 5]

            executor = Mock(side_effect=execute)
            executor.class_obj = env.modules["comfy.ldm.minimax.model"].MiniMaxH3Model()
            wrapped = patches.RefineForward(patches.RefineState(audio=fixed))
            result = wrapped(executor, x, torch.tensor([500.0]), None, options, minimax_payload=payload)
            self.assertEqual(calls, [True])
            self.assertEqual(options, {"minimax_h3_sigma_shift_video": 6.0, "keep": marker})
            self.assertEqual(payload, {"audio_scale": 0.25, "keep": marker})
            self.assertIs(x[1], audio)
            torch.testing.assert_close(audio, audio_before)
            torch.testing.assert_close(fixed, fixed_before)
            torch.testing.assert_close(result[0], video + 1)
            torch.testing.assert_close(result[1], torch.zeros_like(audio))

    def test_forward_failure_leaves_caller_payload_and_audio_unchanged(self):
        with isolated_plugin("patches") as env:
            torch, patches = env.torch, env.patches
            video, audio = torch.zeros((1, 24, 1, 2, 2)), torch.ones((1, 32, 2, 2))
            before = audio.clone()
            options, payload = {"keep": 1}, {"audio_scale": 0.25}
            executor = Mock(side_effect=RuntimeError("预期的前向异常"))
            executor.class_obj = env.modules["comfy.ldm.minimax.model"].MiniMaxH3Model()
            wrapper = patches.RefineForward(patches.RefineState(audio=torch.zeros_like(audio)))
            with self.assertRaisesRegex(RuntimeError, "预期的前向异常"):
                wrapper(executor, [video, audio], torch.tensor([500.0]), None, options, minimax_payload=payload)
            self.assertEqual(options, {"keep": 1})
            self.assertEqual(payload, {"audio_scale": 0.25})
            torch.testing.assert_close(audio, before)

    def test_curve_coordinates_are_added_only_to_forward_copy(self):
        with isolated_plugin("patches") as env:
            torch, patches = env.torch, env.patches
            options, payload = {"keep": 1}, {"keep": 2}
            video, audio = torch.zeros((1, 24, 1, 2, 2)), torch.zeros((1, 32, 2, 1))
            executor = Mock(return_value=[video, audio])
            executor.class_obj = object()
            table = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
            with patch.object(patches, "forward_times", return_value=torch.tensor([0.25])):
                patches.RefineForward(patches.RefineState(table=table))(
                    executor, [video, audio], torch.tensor([500.0]), None, options, minimax_payload=payload,
                )
            copied = executor.call_args.args[3]
            torch.testing.assert_close(copied[patches.COORDS], torch.tensor([[1.0, 3.0]]))
            self.assertEqual(options, {"keep": 1})
            self.assertEqual(payload, {"keep": 2})

    def test_fixed_audio_changes_only_clone_sampling_and_attachments(self):
        with isolated_plugin("patches") as env:
            torch, patches = env.torch, env.patches
            core = env.modules["comfy.ldm.minimax.model"].MiniMaxH3Model()
            model = PatcherDouble(core)
            table = torch.tensor([[0.0], [1.0]])
            original_state = patches.RefineState(table=table)
            original_wrapper = object()
            kind = env.modules["comfy.patcher_extension"].WrappersMP.DIFFUSION_MODEL
            model.attachments[patches.KEY] = original_state
            model.wrappers[(kind, patches.KEY)] = original_wrapper
            audio = torch.full((1, 32, 2, 3), 4.0)
            before = audio.clone()
            result = patches.fixed_audio_model(model, audio)
            self.assertIsNot(result, model)
            self.assertIs(result.model, model.model)
            self.assertIs(model.attachments[patches.KEY], original_state)
            self.assertIs(model.wrappers[(kind, patches.KEY)], original_wrapper)
            self.assertEqual(model.object_patches, {})
            self.assertEqual(model.get_model_object("model_sampling").audio_shift, 3.0)
            self.assertEqual(result.get_model_object("model_sampling").audio_shift, 6.0)
            self.assertIs(result.attachments[patches.KEY].table, table)
            torch.testing.assert_close(result.attachments[patches.KEY].audio, before)
            torch.testing.assert_close(audio, before)
            self.assertIsInstance(result.wrappers[(kind, patches.KEY)], patches.RefineForward)


class SamplingCleanupTests(unittest.TestCase):
    def _run(self, env, fail=False, unload=True, unload_failure=False, sigma_values=(0.7, 0.0)):
        torch, patches, sampling = env.torch, env.patches, env.sampling
        core = env.modules["comfy.ldm.minimax.model"].MiniMaxH3Model()
        model = PatcherDouble(core)
        model.attachments[patches.KEY] = patches.RefineState(audio=torch.zeros((1, 32, 2, 2)))
        initial_shapes = model.model.latent_shapes
        video = torch.ones((1, 24, 1, 2, 2), dtype=torch.float32)
        sigmas = torch.tensor(sigma_values)
        audio_before = model.attachments[patches.KEY].audio.clone()
        metadata = object()
        latent = {"samples": video, "metadata": metadata, "downscale_ratio_temporal": 4}
        noise = SimpleNamespace(seed=123, generate_noise=Mock(return_value=torch.zeros_like(video)))
        mm = env.modules["comfy.model_management"]
        sampler = object()
        case = self

        class Guider:
            def __init__(self, patcher):
                self.model_patcher = patcher

            def set_conds(self, positive, negative):
                pass

            def set_cfg(self, cfg):
                pass

            def sample(self, packed_noise, packed, chosen_sampler, sigmas, **kwargs):
                case.assertIs(chosen_sampler, sampler)
                case.assertEqual(kwargs["seed"], 123)
                torch.testing.assert_close(packed_noise.unbind()[1], torch.zeros_like(audio_before))
                torch.testing.assert_close(packed.unbind()[1], audio_before)
                self.model_patcher.model.latent_shapes = ["采样临时布局"]
                kwargs["callback"](0, None, None, 1)
                if fail:
                    raise RuntimeError("预期的 sampler 异常")
                return packed

        if unload_failure:
            mm.unload_model_and_clones.side_effect = RuntimeError("预期的卸载异常")
        with patch.object(env.modules["comfy.samplers"], "CFGGuider", Guider):
            if fail or unload_failure:
                with self.assertRaisesRegex(RuntimeError, "预期的 sampler 异常" if fail else "预期的卸载异常"):
                    sampling.refine(model, [], latent, sigmas, noise=noise,
                                    sampler=sampler, unload_after=unload)
            else:
                result = sampling.refine(model, [], latent, sigmas, noise=noise,
                                         sampler=sampler, unload_after=unload)
                self.assertIsNot(result, latent)
                self.assertIs(result["metadata"], metadata)
                self.assertNotIn("downscale_ratio_temporal", result)
                torch.testing.assert_close(result["samples"], video)
        self.assertIs(model.model.latent_shapes, initial_shapes)
        self.assertEqual(len(model.clones), 1)
        clone = model.clones[0]
        self.assertEqual(clone.detach_calls, int(unload))
        if unload:
            mm.unload_model_and_clones.assert_called_once_with(clone)
        else:
            mm.unload_model_and_clones.assert_not_called()
        self.assertEqual(latent["downscale_ratio_temporal"], 4)
        self.assertIs(latent["samples"], video)
        torch.testing.assert_close(model.attachments[patches.KEY].audio, audio_before)
        noise.generate_noise.assert_called_once()
        mm.throw_exception_if_processing_interrupted.assert_called_once()

    def test_sampling_restores_shared_layout_after_success_and_failure(self):
        for fail in (False, True):
            for unload in (False, True):
                with self.subTest(fail=fail, unload=unload), isolated_plugin("patches", "sampling") as env:
                    self._run(env, fail=fail, unload=unload)

    def test_detach_and_layout_restore_even_when_unload_raises(self):
        with isolated_plugin("patches", "sampling") as env:
            self._run(env, unload_failure=True)

    def test_external_partial_schedule_can_end_above_zero(self):
        with isolated_plugin("patches", "sampling") as env:
            self._run(env, sigma_values=(0.9, 0.7))

    def test_empty_schedule_returns_video_without_loading_model(self):
        with isolated_plugin("patches", "sampling") as env:
            model = PatcherDouble(env.modules["comfy.ldm.minimax.model"].MiniMaxH3Model())
            video = env.torch.ones((1, 24, 1, 2, 2))
            result = env.sampling.refine(model, [], {"samples": video}, env.torch.tensor([]))
            env.torch.testing.assert_close(result["samples"], video)
            self.assertFalse(model.clones)
            env.modules["comfy.model_management"].unload_model_and_clones.assert_not_called()


class UpscalerIsolationTests(unittest.TestCase):
    def test_targeted_unload_and_detach_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), isolated_plugin("upscaler") as env:
                torch, module = env.torch, env.upscaler
                patcher = SimpleNamespace(model=SimpleNamespace(channels=1), load_device=torch.device("cpu"), detach=Mock())
                upscaler = module.OpenH3Upscaler(patcher, torch.float32, "test.safetensors")
                source = torch.ones((1, 24, 1, 2, 2))
                mask = torch.ones((1, 1, 1, 2, 2))
                latent = {"samples": source, "noise_mask": mask}
                output = torch.ones((1, 24, 1, 4, 4))
                with patch.object(module, "_run_segment", side_effect=RuntimeError("预期异常") if fail else None, return_value=output):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "预期异常"):
                            module.upscale_latent(upscaler, latent, 64, 64, enable_temporal_chunking=False)
                    else:
                        result = module.upscale_latent(upscaler, latent, 64, 64, enable_temporal_chunking=False)
                        self.assertIs(result["samples"], output)
                        self.assertNotIn("noise_mask", result)
                mm = env.modules["comfy.model_management"]
                self.assertIs(mm.load_models_gpu.call_args.args[0][0], patcher)
                mm.unload_model_and_clones.assert_called_once_with(patcher, unload_additional_models=False, all_devices=True)
                patcher.detach.assert_called_once_with()
                self.assertIs(latent["samples"], source)
                self.assertIs(latent["noise_mask"], mask)
