import copy
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
import comfy.patcher_extension
from comfy.ldm.minimax.model import (
    MiniMaxH3Model, PackedLayout, mask_row_values, time_shift_sigma,
    VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP,
)

KEY = "ck_open_h3_regenerate"
COORDS = "ck_open_h3_curve_coords"

CORE = "ck_open_h3_executing_core"

def h3_core(model):
    core = model.get_model_object("diffusion_model")
    if not isinstance(core, MiniMaxH3Model):
        raise TypeError("需要 ComfyUI 原生 MiniMax H3 MODEL；请使用 H3 的模型加载器。")
    return core


def curve_coordinates(table, times):
    position = times.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(-1))


def forward_times(core, x, timestep, options, payload, video_mask=None, audio_mask=None):
    video, audio = x
    t, h, w = video.shape[-3:]
    h, w = (h + 1) // 2 * 2, (w + 1) // 2 * 2
    layout = payload.get("layout")
    if layout is None or layout.signature[1:] != (t, h, w, audio.shape[-1]):
        layout = PackedLayout(0, t, h, w, audio.shape[-1], keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    shift_v = float(options.get("minimax_h3_sigma_shift_video", core.sigma_shift_video))
    shift_a = float(options.get("minimax_h3_sigma_shift_audio", core.sigma_shift_audio))
    tv = float(1.0 - sigma)
    ta = float(1.0 - time_shift_sigma(sigma, shift_v, shift_a))
    visual = max(tv, float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)))
    sound = max(ta, float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)))
    segments = {"text": tv, "video": tv, "audio": ta, "cond": visual,
                "ref_img": visual, "cond_audio": sound, "ref_audio": sound}
    extra = set()
    if video_mask is not None:
        rows = mask_row_values(video_mask[0, 0].float(), t, h, w)
        if rows is not None:
            values = (1.0 - rows * sigma.to(rows.device)).clamp(max=max(tv, VISUAL_COND_TIMESTEP))
            unique = values.unique().tolist()
            if len(unique) == 1:
                segments["video"] = unique[0]
            else:
                extra.update(unique)
    if audio_mask is not None:
        rows = audio_mask[0, 0].float().reshape(-1)
        if not bool((rows >= 1.0 - 1e-3).all()):
            values = (1.0 - rows * (1.0 - ta)).clamp(max=max(ta, AUDIO_COND_TIMESTEP))
            unique = values.unique().tolist()
            if len(unique) == 1:
                segments["audio"] = unique[0]
            else:
                extra.update(unique)
    return torch.tensor(sorted({tv, ta} | {segments[k] for _, _, k in layout.segments} | extra), dtype=torch.float32)


@dataclass(frozen=True)
class RefineState:
    table: object = None
    audio: object = None


class RefineForward:
    def __init__(self, state):
        self.state = state

    def __call__(self, executor, x, timestep, context, transformer_options, **kwargs):
        options = dict(transformer_options)
        payload = dict(kwargs.get("minimax_payload") or {})
        kwargs = dict(kwargs)
        original_audio = x[1]
        core = executor.class_obj
        options[CORE] = core
        if self.state.audio is not None:
            fixed = self.state.audio.to(device=x[0].device, dtype=x[0].dtype)
            x = [x[0], fixed]
            payload["audio_scale"] = 1.0
            options["minimax_h3_sigma_shift_audio"] = float(options.get("minimax_h3_sigma_shift_video", core.sigma_shift_video))
            kwargs["audio_denoise_mask"] = fixed.new_zeros((1, 1, fixed.shape[2], fixed.shape[3]))
        kwargs["minimax_payload"] = payload
        if self.state.table is not None:
            times = forward_times(core, x, timestep, options, payload, kwargs.get("denoise_mask"), kwargs.get("audio_denoise_mask"))
            options[COORDS] = curve_coordinates(self.state.table, times)
        result = executor(x, timestep, context, options, **kwargs)
        if self.state.audio is not None:
            return [result[0], torch.zeros_like(original_audio)]
        return result


def install_forward(model, state):
    model.set_attachments(KEY, state)
    kind = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    model.remove_wrappers_with_key(kind, KEY)
    model.add_wrapper_with_key(kind, KEY, RefineForward(state))


def fixed_audio_model(model, audio):
    h3_core(model)
    if audio.ndim != 4 or audio.shape[:3] != (1, 32, 2):
        raise ValueError("静音条件需要 H3 音频 latent [1,32,2,T]。")
    patched = model.clone()
    state = patched.get_attachment(KEY) or RefineState()
    sampling = copy.deepcopy(patched.get_model_object("model_sampling"))
    sampling.audio_shift = sampling.shift
    patched.add_object_patch("model_sampling", sampling)
    install_forward(patched, replace(state, audio=audio.detach().cpu().float()))
    return patched


class SharpBlockPatch:
    def __init__(self, index, a, b, strength, previous=None):
        self.index = index
        self.a = a.detach().cpu().float()
        self.b = b.detach().cpu().float()
        self.strength = strength
        self.previous = previous

    def __call__(self, args, extra):
        coords = args["transformer_options"][COORDS]
        if coords.shape[0] != args["t_emb"].shape[0]:
            raise ValueError("Sharp 时间行与当前 H3 前向不匹配；请检查其他模型补丁是否改变了时间布局。")
        delta = F.linear(F.linear(coords, self.a), self.b) * self.strength

        def add_curve(module, inputs, output):
            return output + delta.to(device=output.device, dtype=output.dtype)

        linear = args["transformer_options"][CORE].blocks[self.index].adaln_proj.linear
        handle = linear.register_forward_hook(add_curve)
        try:
            if self.previous is not None:
                return self.previous(args, extra)
            return extra["original_block"](args)
        finally:
            handle.remove()
