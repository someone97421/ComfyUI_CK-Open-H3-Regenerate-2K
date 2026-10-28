import logging
import re
from dataclasses import replace

import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter

from .patches import KEY, RefineState, SharpBlockPatch, h3_core, install_forward
from .weights import load_curve_table, resolve_file


def native_name(name):
    name = name.removeprefix("transformer.")
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    if name == "norm_out.linear":
        name = "final_layer.adaln_proj.linear"
    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks."):]
    return name


def mapped_parts(name, a, b):
    """Diffusers 分离投影映射到原生融合投影，以切片避免拼接完整 QKV。"""
    name = native_name(name)
    match = re.search(r"\.attn\.to_([qkv])$", name)
    if match:
        part = {"q": 0, "k": 1, "v": 2}[match[1]]
        target = name[:match.start()] + ".attn.qkv_proj"
        return [(target, (0, part * b.shape[0], b.shape[0]), a, b)]
    if name.endswith(".attn.to_out.0"):
        name = name[:-len(".attn.to_out.0")] + ".attn.out_proj"
    elif name.endswith(".ff.net.0.proj"):
        target = name[:-len(".ff.net.0.proj")] + ".mlp.fc1"
        if b.shape[0] % 2:
            raise ValueError(f"MLP 输出行数不是偶数：{name}")
        half = b.shape[0] // 2
        return [(target, (0, 0, half), a, b[half:]), (target, (0, half, half), a, b[:half])]
    elif name.endswith(".ff.net.2"):
        name = name[:-len(".ff.net.2")] + ".mlp.fc2"
    return [(name, None, a, b)]


def apply_regenerate(model, lora_path, curve_path, curve_table, lora_strength=1.0, curve_strength=1.0):
    core = h3_core(model)
    patched = model.clone()
    count = 0
    if lora_strength != 0:
        raw = comfy.utils.load_torch_file(resolve_file(lora_path, ("loras",)), safe_load=True)
        roots = sorted(k[:-len(".lora_A.weight")] for k in raw if k.endswith(".lora_A.weight"))
        if not roots:
            raise ValueError("文件中没有 .lora_A.weight；此节点读取 Open-H3 发布的合并 LoRA。普通 H3 LoRA 可使用原生 LoRA 节点。")
        patches = {}
        used = set()
        for root in roots:
            ak, bk = root + ".lora_A.weight", root + ".lora_B.weight"
            a, b = raw[ak], raw[bk]
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError(f"LoRA A/B 形状不匹配：{root}")
            for name, offset, down, up in mapped_parts(root, a, b):
                layer = core.get_submodule(name)
                rows, columns = layer.weight.shape
                if offset is not None:
                    source_rows = 3 * b.shape[0] if name.endswith(".attn.qkv_proj") else b.shape[0]
                    if source_rows != rows:
                        raise ValueError(f"融合投影行数不匹配：{name}，模型 {rows} 行，LoRA 需要 {source_rows} 行。")
                expected_rows = rows if offset is None else offset[2]
                if columns != down.shape[1] or expected_rows != up.shape[0] or (offset is not None and offset[1] + offset[2] > rows):
                    raise ValueError(f"LoRA 与基础模型不匹配：{name}，模型 {tuple(layer.weight.shape)}，A {tuple(down.shape)}，B {tuple(up.shape)}")
                key = "diffusion_model." + name + ".weight"
                if offset is not None:
                    key = (key, offset)
                patches[key] = LoRAAdapter(set(), (up, down, None, None, None, None))
            used.update((ak, bk))
        unused = set(raw) - used
        if unused:
            logging.warning("[OpenH3] 合并 LoRA 有 %d 个非 A/B 字段未使用：%s", len(unused), sorted(unused)[:8])
        accepted = patched.add_patches(patches, strength_patch=lora_strength)
        if len(accepted) != len(patches):
            raise ValueError("部分 LoRA 投影未被模型接受，无法完整应用 Refiner。")
        count = len(roots)
        del raw

    curve_count = 0
    if curve_strength != 0:
        raw = comfy.utils.load_torch_file(resolve_file(curve_path, ("loras",)), safe_load=True)
        roots = sorted(k[:-len(".curve_A.weight")] for k in raw if k.endswith(".curve_A.weight"))
        if not roots:
            raise ValueError("Sharp 文件中没有 curve_A.weight；请使用 sharp_curve_branches.safetensors。")
        table = load_curve_table(patched, curve_table)
        old = patched.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for root in roots:
            name = native_name(root)
            match = re.fullmatch(r"blocks\.(\d+)\.adaln_proj\.linear", name)
            if match is None:
                raise ValueError(f"无法映射 Sharp 分支：{root}")
            index = int(match[1])
            block = core.blocks[index]
            a, b = raw[root + ".curve_A.weight"], raw[root + ".curve_B.weight"]
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 8 or b.shape != (block.adaln_proj.linear.weight.shape[0], a.shape[0]):
                raise ValueError(f"Sharp 分支形状与 H3 block 不匹配：{name}")
            patch = SharpBlockPatch(index, a, b, curve_strength, old.get(("double_block", index)))
            patched.set_model_patch_replace(patch, "dit", "double_block", index)
        curve_count = len(roots)
        if curve_count != len(core.blocks):
            logging.warning("[OpenH3] 本次应用 %d/%d 个 Sharp block 分支。", curve_count, len(core.blocks))
        state = patched.get_attachment(KEY) or RefineState()
        install_forward(patched, replace(state, table=table))
        del raw
    report = f"已应用 {count} 组 LoRA 投影、{curve_count} 个 Sharp 曲线分支；基础模型由 ComfyUI 共享管理。"
    logging.info("[OpenH3] %s", report)
    return patched, report
