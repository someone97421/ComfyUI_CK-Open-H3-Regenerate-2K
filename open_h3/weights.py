import hashlib
import logging
import os
from pathlib import Path

import folder_paths
import torch
from safetensors import safe_open

CURVE_SHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"


def resolve_file(name, categories):
    path = Path(os.path.expandvars(os.path.expanduser(name.strip().strip('"'))))
    if path.is_file():
        return str(path.resolve())
    for category in categories:
        if category in folder_paths.folder_names_and_paths:
            found = folder_paths.get_full_path(category, str(path))
            if found and Path(found).is_file():
                return found
        candidate = Path(folder_paths.models_dir) / category / path
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"找不到模型文件：{name}。可填写绝对路径或模型目录中的相对路径。")


def model_files(category):
    if category not in folder_paths.folder_names_and_paths:
        return []
    return folder_paths.get_filename_list(category)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().float().contiguous().numpy().tobytes()).hexdigest()


def read_curve_file(path):
    if Path(path).suffix.lower() == ".bin":
        raw = Path(path).read_bytes()
        if len(raw) != 1025 * 8 * 4:
            raise ValueError("曲线 bin 文件应为 1025×8 个 FP32 数值，共 32800 字节。")
        return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(1025, 8).clone()
    with safe_open(path, framework="pt", device="cpu") as reader:
        for key in ("adaln_t_table", "diffusion_model.adaln_t_table", "model.diffusion_model.adaln_t_table"):
            if key in reader.keys():
                table = reader.get_tensor(key).float()
                if tuple(table.shape) != (1025, 8):
                    raise ValueError(f"曲线表形状不匹配：{tuple(table.shape)}")
                return table
    raise KeyError(f"文件中没有 adaln_t_table：{path}")


def load_curve_table(model, source="auto"):
    if source.strip() and source.strip().lower() != "auto":
        path = resolve_file(source, ("diffusion_models", "loras"))
        table = read_curve_file(path)
        if tensor_digest(table) != CURVE_SHA256:
            logging.warning("[OpenH3] 使用用户指定曲线表；其哈希与 NVIDIA 参考表不同。")
        return table
    core = model.get_model_object("diffusion_model")
    native = getattr(core, "adaln_t_table", None)
    if native is not None and native.device.type != "meta" and tensor_digest(native) == CURVE_SHA256:
        return native.detach().cpu().float().clone()
    candidates = [n for n in model_files("diffusion_models") if "h3" in n.lower() and "ref2va" in n.lower() and n.endswith(".safetensors")]
    for name in candidates:
        path = resolve_file(name, ("diffusion_models",))
        try:
            table = read_curve_file(path)
        except KeyError:
            continue
        if tensor_digest(table) == CURVE_SHA256:
            logging.info("[OpenH3] 从本地 %s 读取 32KB Sharp 曲线表。", name)
            return table
    raise FileNotFoundError("未找到参考 Sharp 曲线表。curve_table 填写含 adaln_t_table 的 REF2VA safetensors 路径或 adaln_t_table.bin；不会自动下载。")
