"""OpenH3 潜空间放大网络与 ComfyUI 模型管理适配。

网络改编自 LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler 的
nodes/minimax_h3_latent_upscaler_3d.py，许可及来源见
../licenses/latent-upscaler-MIT.txt。
"""

import logging
import os
import re
import threading
from collections.abc import Mapping
import traceback

import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
import folder_paths


VAE_DOWNSAMPLE = 16
MODEL_FOLDER = "latent_upscale_models"
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = normalization(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(1, 2).unsqueeze(1)
        k = self.k(h).flatten(2).transpose(1, 2).unsqueeze(1)
        v = self.v(h).flatten(2).transpose(1, 2).unsqueeze(1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).transpose(1, 2).reshape(x.shape)
        return x + self.proj_out(h)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0), groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


def _block_stack(sd, prefix, channels):
    indices = sorted({int(match.group(1)) for key in sd
                      if (match := re.match(rf"{prefix}\.(\d+)\.", key))})
    if not indices or indices != list(range(len(indices))):
        raise ValueError(f"权重中的 {prefix} 模块编号必须从 0 连续排列。")
    blocks = nn.ModuleList()
    for index in indices:
        base = f"{prefix}.{index}."
        if base + "in_layers.0.weight" in sd:
            block = ResBlockEmb3D(channels, 64, dropout=0.1)
        elif base + "dwconv.weight" in sd:
            kernel = sd[base + "dwconv.weight"].shape[2]
            if kernel < 1 or kernel % 2 != 1:
                raise ValueError("时间卷积核必须为正奇数。")
            block = TemporalConv(channels, kernel)
        elif base + "q.weight" in sd:
            block = AttnBlock3D(channels)
        else:
            raise ValueError(f"无法识别权重模块：{prefix}.{index}。")
        blocks.append(block)
    return blocks


class LatentResizer3D(nn.Module):
    def __init__(self, sd):
        super().__init__()
        channels, in_channels = sd["conv_in.weight"].shape[:2]
        if in_channels != 24 or channels % 32:
            raise ValueError("仅支持 24 通道 OpenH3 权重，网络宽度必须是 32 的倍数。")
        self.channels = channels
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64),
        )
        self.in_blocks = _block_stack(sd, "in_blocks", channels)
        self.out_blocks = _block_stack(sd, "out_blocks", channels)
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def forward(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device,
        ).unsqueeze(0)
        emb = self.embed(scale_emb)
        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                emb_t = emb.expand(x.shape[0], -1)
                x = b(x, emb_t)
            else:
                x = b(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


def resolve_model_path(model_name):
    name = os.path.expanduser(model_name.strip())
    if not name:
        raise ValueError("请填写权重绝对路径，或 latent_upscale_models 目录中的相对文件名。")
    path = name if os.path.isabs(name) else folder_paths.get_full_path(MODEL_FOLDER, name)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 OpenH3 放大权重：{name}。")
    return os.path.abspath(path)


def _extract_state_dict(raw):
    while isinstance(raw, Mapping):
        nested = next((raw[key] for key in ("state_dict", "model")
                       if isinstance(raw.get(key), Mapping)), None)
        if nested is None:
            break
        raw = nested
    if not isinstance(raw, Mapping) or not all(isinstance(key, str) for key in raw):
        raise ValueError("权重必须包含可识别的张量 state_dict。")
    if any(key.startswith("upscaler.") for key in raw):
        raw = {key[len("upscaler."):]: value for key, value in raw.items()
               if key.startswith("upscaler.")}
    if "conv_in.weight" not in raw or not all(isinstance(value, torch.Tensor) for value in raw.values()):
        raise ValueError("未找到有效的 OpenH3 放大网络权重；支持裸 state_dict 或 upscaler. 前缀。")
    return dict(raw)


class _FullLoadModelPatcher(comfy.model_patcher.ModelPatcher):
    """原生 torch 层需要完整权重；包括 ComfyUI 的 --novram 模式。"""

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        return super().load(device_to, lowvram_model_memory=0,
                            force_patch_weights=force_patch_weights, full_load=True)


class OpenH3Upscaler:
    """同一节点输出共享一个 patcher；锁覆盖加载、推理与卸载。"""

    def __init__(self, patcher, dtype, path):
        self.patcher = patcher
        self.dtype = dtype
        self.path = path
        self.lock = threading.RLock()


def load_upscaler(model_name, precision="auto"):
    path = resolve_model_path(model_name)
    try:
        sd = _extract_state_dict(comfy.utils.load_torch_file(path, safe_load=True))
        device = mm.get_torch_device()
        supported = (torch.float16, torch.bfloat16, torch.float32)
        if precision == "auto":
            dtype = mm.unet_dtype(
                device=device, model_params=sum(value.numel() for value in sd.values()),
                supported_dtypes=list(supported),
            )
            if dtype not in supported or device.type == "cpu":
                dtype = torch.float32
        else:
            dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
            if dtype is None:
                raise ValueError(f"不支持的计算精度：{precision}。")
        with torch.device("meta"):
            model = LatentResizer3D(sd)
        model.load_state_dict(sd, strict=True, assign=True)
        del sd
        model = model.to(device="cpu", dtype=dtype).eval().requires_grad_(False)
        patcher = _FullLoadModelPatcher(
            model, load_device=device, offload_device=torch.device("cpu"),
        )
        return OpenH3Upscaler(patcher, dtype, path)
    except Exception as exc:
        raise RuntimeError(f"加载 OpenH3 放大权重失败：{exc}") from exc


def _validate_samples(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("请输入包含 samples 的标准 LATENT。")
    samples = latent["samples"]
    if getattr(samples, "is_nested", False) or isinstance(samples, (list, tuple)):
        raise ValueError("不支持 nested LATENT；请先拆分为单个视频 LATENT 后分别放大。")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        raise ValueError("仅支持视频 LATENT 张量 [B,24,T,H,W]；nested 输入请先拆分视频。")
    if samples.shape[1] != 24 or any(size < 1 for size in samples.shape):
        raise ValueError("视频 LATENT 必须为非空 [B,24,T,H,W]，通道数必须为 24。")
    if not samples.is_floating_point():
        raise ValueError("LATENT samples 必须为浮点张量。")
    return samples


def _target_size(samples, width, height, align):
    if width < 16 or height < 16 or align < 1:
        raise ValueError("目标宽高至少为 16 像素，align 必须为正整数。")
    h_in, w_in = samples.shape[-2:]
    effective_scale = (width / (w_in * VAE_DOWNSAMPLE) + height / (h_in * VAE_DOWNSAMPLE)) / 2.0
    w_aligned = round(width / align) * align
    h_aligned = round(height / align) * align
    w_out = max(1, int(round(w_aligned / VAE_DOWNSAMPLE)))
    h_out = max(1, int(round(h_aligned / VAE_DOWNSAMPLE)))
    if effective_scale < 1.0 and (w_out < w_in or h_out < h_in):
        raise ValueError("此模型的请求平均倍率小于 1，且对齐后存在缩小轴；请增大目标尺寸。")
    return h_out, w_out, effective_scale


def _run_segment(model, samples, scale, height, width, device, dtype, output_dtype):
    mm.throw_exception_if_processing_interrupted()
    x = samples.to(device=device, dtype=dtype).contiguous()
    mean = torch.tensor(LATENTS_MEAN, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=device, dtype=dtype).view(1, 24, 1, 1, 1)
    x = (x - mean) / std
    x = model(x, scale=scale, size=(x.shape[2], height, width))
    x = x * std + mean
    return x.to(device="cpu", dtype=output_dtype)


def _run_chunked(model, samples, scale, height, width, chunk_size, overlap, device, dtype):
    batch, channels, frames = samples.shape[:3]
    output = torch.zeros((batch, channels, frames, height, width), dtype=torch.float32, device="cpu")
    weights = torch.zeros((1, 1, frames, 1, 1), dtype=torch.float32, device="cpu")
    for start in range(0, frames, chunk_size):
        mm.throw_exception_if_processing_interrupted()
        end = min(frames, start + chunk_size)
        out_start, out_end = max(0, start - overlap), min(frames, end + overlap)
        lo, hi = out_start - overlap, out_end + overlap
        segment = samples[:, :, max(0, lo):min(frames, hi)].to(device=device, dtype=dtype).contiguous()
        if lo < 0 or hi > frames:
            segment = F.pad(segment, (0, 0, 0, 0, max(0, -lo), max(0, hi - frames)), mode="replicate")
        predicted = _run_segment(model, segment, scale, height, width, device, dtype, torch.float32)
        valid = predicted[:, :, overlap:overlap + out_end - out_start]
        weight = torch.ones(out_end - out_start, dtype=torch.float32, device="cpu")
        left, right = start - out_start, out_end - end
        if left:
            weight[:left] = torch.arange(1, left + 1, dtype=torch.float32) / (left + 1)
        if right:
            weight[-right:] = torch.arange(right, 0, -1, dtype=torch.float32) / (right + 1)
        weight = weight.view(1, 1, -1, 1, 1)
        output[:, :, out_start:out_end].add_(valid * weight)
        weights[:, :, out_start:out_end].add_(weight)
        del segment, predicted, valid, weight
    output.div_(weights.clamp(min=1e-8))
    return output.to(dtype=samples.dtype)


def upscale_latent(upscaler, latent, width, height, align=32, enable_temporal_chunking=True,
                   chunk_size=32, overlap=5):
    if not isinstance(upscaler, OpenH3Upscaler):
        raise ValueError("请连接 OpenH3UpscalerLoader 输出的 OPEN_H3_UPSCALER。")
    samples = _validate_samples(latent)
    height_out, width_out, scale = _target_size(samples, width, height, align)
    if chunk_size < 1 or overlap < 0:
        raise ValueError("chunk_size 必须为正整数，overlap 不能为负数。")
    result = latent.copy()
    if (height_out, width_out) == tuple(samples.shape[-2:]):
        return result
    chunked = enable_temporal_chunking and samples.shape[2] > chunk_size
    if chunked:
        logging.info("[OpenH3] 时间分块为近似模式：GroupNorm/注意力使用块内统计，结果可能与整段推理不同。")
    patcher = upscaler.patcher
    frames = min(samples.shape[2] + 2 * overlap, chunk_size + 4 * overlap) if chunked else samples.shape[2]
    activation_memory = (samples.shape[0] * patcher.model.channels * frames
                         * (samples.shape[-2] * samples.shape[-1] + height_out * width_out)
                         * mm.dtype_size(upscaler.dtype) * 8)
    with upscaler.lock:
        try:
            # 原生 nn 层无逐层权重转换，必须请求完整加载。
            mm.load_models_gpu([patcher], memory_required=activation_memory, force_full_load=True)
            with torch.inference_mode():
                if chunked:
                    output = _run_chunked(
                        patcher.model, samples, scale, height_out, width_out,
                        chunk_size, overlap, patcher.load_device, upscaler.dtype,
                    )
                else:
                    output = _run_segment(
                        patcher.model, samples, scale, height_out, width_out,
                        patcher.load_device, upscaler.dtype, samples.dtype,
                    )
            result["samples"] = output
        except Exception as exc:
            traceback.clear_frames(exc.__traceback__)
            raise RuntimeError(f"OpenH3 潜空间放大失败；可尝试减小目标尺寸、chunk_size 或 overlap：{exc}") from exc
        except BaseException as exc:
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            try:
                mm.unload_model_and_clones(patcher, unload_additional_models=False, all_devices=True)
            finally:
                patcher.detach()
    if "noise_mask" in result:
        result.pop("noise_mask")
        logging.warning("[OpenH3] 已移除 noise_mask：学习式放大会改变潜空间网格，原掩码不能直接复用；请在放大后重新设置噪声掩码。")
    return result
