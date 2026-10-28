import torch

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy.ldm.minimax.model import unpatchify_video

from .patches import KEY, fixed_audio_model, h3_core


def video_tensor(latent):
    tensor = latent["samples"]
    if tensor.is_nested:
        tensor = tensor.unbind()[0]
    if tensor.ndim != 5 or tensor.shape[0] != 1 or tensor.shape[1] != 24 or min(tensor.shape) < 1:
        raise ValueError("H3 精炼需要视频 LATENT [1,24,T,H,W]；音视频组合 LATENT 会自动取视频流。")
    return tensor


def video_frames(tokens):
    return (tokens // 5) * 17 + sum((1, 4, 4, 4, 4)[:tokens % 5])


def make_noise(video, seed, mode):
    device = mm.get_torch_device() if mode == "reference_device" else torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    b, c, t, h, w = video.shape
    ph, pw = (h + 1) // 2, (w + 1) // 2
    rows = torch.randn((b, t * ph * pw, c * 4), dtype=torch.float32, device=device, generator=generator)
    return unpatchify_video(rows, t, ph, pw, c)[:, :, :, :h, :w].cpu()


def refine(model, positive, latent, sigmas, seed=0, cfg=1.0, negative=None,
           noise=None, sampler=None, audio_latent=None, noise_mode="reference_device", unload_after=True):
    h3_core(model)
    video = video_tensor(latent).detach().cpu().float()
    if sigmas.ndim != 1:
        raise ValueError("SIGMAS 必须是一维张量。")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("SIGMAS 不能包含 NaN 或无穷值。")
    if sigmas.numel() < 2:
        result = dict(latent)
        result["samples"] = video
        if "noise_mask" in result and result["noise_mask"].is_nested:
            result["noise_mask"] = result["noise_mask"].unbind()[0]
        return result
    if audio_latent is not None:
        model = fixed_audio_model(model, audio_latent["samples"])
    else:
        model = model.clone()
    state = model.get_attachment(KEY)
    if state is None or state.audio is None:
        raise ValueError("请先连接 OpenH3 Refine Context，或给 audio_latent 输入连接 H3 静音编码 LATENT。")
    if negative is None and cfg != 1.0:
        raise ValueError("CFG 不为 1 时需要连接 negative CONDITIONING。")
    audio = state.audio
    if noise is None:
        video_noise = make_noise(video, seed, noise_mode)
    else:
        video_noise = noise.generate_noise({**latent, "samples": video})
        seed = getattr(noise, "seed", seed)
        if video_noise.is_nested:
            video_noise = video_noise.unbind()[0]
        if tuple(video_noise.shape) != tuple(video.shape):
            raise ValueError("外接 NOISE 的视频张量形状与输入 LATENT 不一致。")
        video_noise = video_noise.cpu().float()
    packed = comfy.nested_tensor.NestedTensor((video, audio))
    packed_noise = comfy.nested_tensor.NestedTensor((video_noise, torch.zeros_like(audio)))
    mask = latent.get("noise_mask")
    if mask is not None:
        if mask.is_nested:
            mask = mask.unbind()[0]
        mask = comfy.nested_tensor.NestedTensor((mask, torch.ones_like(audio)))
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative if negative is not None else positive)
    guider.set_cfg(cfg)
    sampler = sampler if sampler is not None else comfy.samplers.sampler_object("euler")
    progress = comfy.utils.ProgressBar(len(sigmas) - 1)

    def callback(step, x0, x, total):
        mm.throw_exception_if_processing_interrupted()
        progress.update_absolute(step + 1, total)

    previous_shapes = model.model.latent_shapes
    try:
        sampled = guider.sample(packed_noise, packed, sampler, sigmas, denoise_mask=mask,
                                callback=callback, disable_pbar=False, seed=seed)
        result = dict(latent)
        result["samples"] = sampled.unbind()[0].to(device=mm.intermediate_device(), dtype=torch.float32)
        if "noise_mask" in result and result["noise_mask"].is_nested:
            result["noise_mask"] = result["noise_mask"].unbind()[0]
        result.pop("downscale_ratio_spacial", None)
        result.pop("downscale_ratio_temporal", None)
        return result
    finally:
        model.model.latent_shapes = previous_shapes
        if unload_after:
            try:
                mm.unload_model_and_clones(guider.model_patcher)
            finally:
                guider.model_patcher.detach()
