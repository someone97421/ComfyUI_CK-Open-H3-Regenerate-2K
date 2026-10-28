CATEGORY = "CK/Open-H3-Regenerate"


class OpenH3ApplyRegenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_path": ("STRING", {"default": "Open-H3-Regenerate-2K/open_h3_regenerate_2k_lora.safetensors", "tooltip": "绝对路径或 models/loras 下的相对路径；发布版已合并 Refiner、Turbo 和 Sharp 线性增量。"}),
            "sharp_path": ("STRING", {"default": "Open-H3-Regenerate-2K/sharp_curve_branches.safetensors"}),
            "curve_table": ("STRING", {"default": "auto", "tooltip": "auto 从本地 REF2VA 模型读取参考曲线表；也可指定 safetensors 或 bin 路径。"}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            "curve_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "Sharp 时间曲线分支的独立强度；设为 0 可单独研究线性 LoRA。"}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY

    def apply(self, model, lora_path, sharp_path, curve_table, lora_strength, curve_strength):
        from .open_h3.adapters import apply_regenerate
        return apply_regenerate(model, lora_path, sharp_path, curve_table, lora_strength, curve_strength)


class OpenH3RefineContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "conditioning": ("CONDITIONING",),
            "audio_latent": ("LATENT", {"tooltip": "参考流程使用由 H3 音频 VAE 编码的静音，可接 OpenH3 Encode Silence。"}),
            "clear_reference_guides": ("BOOLEAN", {"default": True, "tooltip": "参考精炼只保留文本条件。关闭后保留原有 H3 首尾帧/参考条件，供实验使用。"}),
        }}

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, model, conditioning, audio_latent, clear_reference_guides):
        from .open_h3.patches import fixed_audio_model
        model = fixed_audio_model(model, audio_latent["samples"])
        clean = []
        for embedding, metadata in conditioning:
            metadata = dict(metadata)
            if clear_reference_guides:
                for key in ("minimax_keyframes", "minimax_refs", "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug"):
                    metadata.pop(key, None)
            clean.append([embedding, metadata])
        return model, clean


class OpenH3SigmaSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (["reference_3", "reference_2", "shifted", "custom"],),
            "steps": ("INT", {"default": 3, "min": 1, "max": 1000}),
            "start_sigma": ("FLOAT", {"default": 0.909375, "min": 0.000001, "max": 1.0, "step": 0.001}),
            "shift": ("FLOAT", {"default": 6.0, "min": 0.001, "max": 1000.0}),
            "custom_sigmas": ("STRING", {"default": "0.909375, 0.8571428571428571, 0.6666666666666666, 0", "tooltip": "仅 custom 模式读取；reference 模式使用原始固定调度，忽略步数和起始值。"}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, preset, steps, start_sigma, shift, custom_sigmas):
        import torch
        from .open_h3.schedules import schedule_values
        return (torch.tensor(schedule_values(preset, steps, start_sigma, shift, custom_sigmas), dtype=torch.float32),)


class OpenH3RefineSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "latent": ("LATENT",),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 1766580932, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "noise_mode": (["reference_device", "cpu"], {"tooltip": "使用独立 RNG 按 H3 patch 顺序生成噪声；外接 NOISE 时忽略此选项和 seed。"}),
            "unload_after": ("BOOLEAN", {"default": True, "tooltip": "结束或异常时卸载本次 H3 模型，给 VAE 腾出显存。关闭可保留 ComfyUI 模型缓存。"}),
        }, "optional": {
            "negative": ("CONDITIONING",),
            "noise": ("NOISE",),
            "sampler": ("SAMPLER", {"tooltip": "不连接时使用原生 Euler；其他采样器用于实验。"}),
            "audio_latent": ("LATENT", {"tooltip": "可直接提供固定音频条件，或使用 Refine Context 节点。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("refined_video",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent, sigmas, seed, cfg, noise_mode, unload_after,
               negative=None, noise=None, sampler=None, audio_latent=None):
        from .open_h3.sampling import refine
        return (refine(model, positive, latent, sigmas, seed, cfg, negative,
                       noise, sampler, audio_latent, noise_mode, unload_after),)
