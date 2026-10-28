"""受控宿主协议替身：仅隔离导入与记录调用，不模拟 ComfyUI 的计算和显存管理。"""

import copy
import importlib
import importlib.util
from contextlib import contextmanager
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import uuid


ROOT = Path(__file__).resolve().parents[1]


def _unexpected(*args, **kwargs):
    raise AssertionError("此测试不允许调用真实宿主计算、模型扫描或权重加载。")


def _host_modules(torch):
    modules = {}

    def module(name, package=False):
        result = ModuleType(name)
        result.__spec__ = ModuleSpec(name, loader=None, is_package=package)
        if package:
            result.__path__ = []
        modules[name] = result
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, result)
        return result

    module("comfy", package=True)
    extension = module("comfy.patcher_extension")
    extension.WrappersMP = SimpleNamespace(DIFFUSION_MODEL="diffusion_model")
    module("comfy.model_patcher").ModelPatcher = type("ModelPatcher", (), {})
    module("comfy.ldm", package=True)
    module("comfy.ldm.minimax", package=True)
    minimax = module("comfy.ldm.minimax.model")

    class MiniMaxH3Model:
        sigma_shift_video = 6.0
        sigma_shift_audio = 3.0

    minimax.MiniMaxH3Model = MiniMaxH3Model
    minimax.PackedLayout = _unexpected
    minimax.mask_row_values = _unexpected
    minimax.time_shift_sigma = _unexpected
    minimax.unpatchify_video = _unexpected
    minimax.VISUAL_COND_TIMESTEP = 0.999
    minimax.AUDIO_COND_TIMESTEP = 1.0

    mm = module("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.intermediate_device = lambda: torch.device("cpu")
    mm.throw_exception_if_processing_interrupted = Mock()
    mm.unload_model_and_clones = Mock()
    mm.load_models_gpu = Mock()
    mm.dtype_size = lambda dtype: torch.empty((), dtype=dtype).element_size()
    nested = module("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True

        def __init__(self, tensors):
            self.tensors = tuple(tensors)

        def unbind(self):
            return self.tensors

    nested.NestedTensor = NestedTensor
    samplers = module("comfy.samplers")
    samplers.CFGGuider = _unexpected
    samplers.sampler_object = _unexpected
    utils = module("comfy.utils")
    utils.load_torch_file = _unexpected
    utils.ProgressBar = lambda total: SimpleNamespace(update_absolute=Mock())
    module("comfy.weight_adapter", package=True)
    module("comfy.weight_adapter.lora").LoRAAdapter = _unexpected
    paths = module("folder_paths")
    paths.folder_names_and_paths = {}
    paths.models_dir = str(ROOT / "__test_models_not_present__")
    paths.get_filename_list = _unexpected
    paths.get_full_path = _unexpected
    return modules


@contextmanager
def isolated_plugin(*names):
    """加载真实实现；torch/safetensors 保持真实，仅替换宿主及本次私有包名。"""
    torch = importlib.import_module("torch")
    prefix = "_ck_open_h3_test_" + uuid.uuid4().hex
    roots = ("comfy", "folder_paths", prefix)

    def owned(name):
        return any(name == root or name.startswith(root + ".") for root in roots)

    previous = {name: value for name, value in tuple(sys.modules.items()) if owned(name)}
    stubs = _host_modules(torch)
    try:
        for name in previous:
            del sys.modules[name]
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(
            prefix, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[prefix] = package
        spec.loader.exec_module(package)
        loaded = {name: importlib.import_module(prefix + ".open_h3." + name) for name in names}
        yield SimpleNamespace(torch=torch, modules=stubs, package=package, **loaded)
    finally:
        for name in tuple(sys.modules):
            if owned(name):
                del sys.modules[name]
        sys.modules.update(previous)


class PatcherDouble:
    """最小可观测模型协议，clone 共享底层 model、独立保存补丁容器。"""

    def __init__(self, core):
        self.objects = {
            "diffusion_model": core,
            "model_sampling": SimpleNamespace(shift=6.0, audio_shift=3.0),
        }
        self.model = SimpleNamespace(latent_shapes=["原始布局"])
        self.attachments = {}
        self.wrappers = {}
        self.object_patches = {}
        self.clones = []
        self.detach_calls = 0

    def clone(self):
        result = copy.copy(self)
        result.attachments = self.attachments.copy()
        result.wrappers = self.wrappers.copy()
        result.object_patches = self.object_patches.copy()
        result.detach_calls = 0
        self.clones.append(result)
        return result

    def get_model_object(self, name):
        return self.object_patches.get(name, self.objects[name])

    def get_attachment(self, key):
        return self.attachments.get(key)

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.pop((kind, key), None)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers[(kind, key)] = wrapper

    def add_object_patch(self, name, value):
        self.object_patches[name] = value

    def detach(self):
        self.detach_calls += 1
