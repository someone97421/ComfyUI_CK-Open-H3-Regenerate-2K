"""OpenH3 潜空间放大的传统 ComfyUI 节点。"""


class OpenH3UpscalerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {
                    "default": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                    "tooltip": "权重的本地绝对路径，或现有 latent_upscale_models 目录中的相对文件名（支持子目录）。不下载权重。",
                }),
                "precision": (["auto", "fp32", "fp16", "bf16"], {
                    "default": "auto",
                    "tooltip": "auto 由 ComfyUI 选择计算精度，CPU 自动使用 fp32。模型先加载到 CPU，推理时由 ComfyUI 调度。",
                }),
            },
        }

    RETURN_TYPES = ("OPEN_H3_UPSCALER",)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "load"
    CATEGORY = "CK/Open-H3-Regenerate"
    DESCRIPTION = "加载 OpenH3 放大权重；输出可供多个放大节点复用的模型。"

    def load(self, model_name, precision="auto"):
        from .open_h3.upscaler import load_upscaler
        return (load_upscaler(model_name, precision),)


class OpenH3LatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaler": ("OPEN_H3_UPSCALER", {
                    "tooltip": "连接 OpenH3UpscalerLoader。完整模型权重需要能装入计算设备；每次推理结束或异常时卸载回 CPU。",
                }),
                "latent": ("LATENT", {
                    "tooltip": "标准视频 LATENT，samples 为 [B,24,T,H,W]。nested LATENT 请先拆分视频。保留其他元数据；实际放大时移除 noise_mask，请在输出后重新设置。",
                }),
                "width": ("INT", {
                    "default": 2560, "min": 16, "max": 16384, "step": 16,
                    "tooltip": "目标像素宽度；先按 align 就近取整，再按 VAE 的 16 倍下采样取整。仅支持放大。",
                }),
                "height": ("INT", {
                    "default": 1440, "min": 16, "max": 16384, "step": 16,
                    "tooltip": "目标像素高度；时间长度保持不变。遵循原节点的平均倍率判定，可在放大时调整宽高比例。",
                }),
                "align": ("INT", {
                    "default": 32, "min": 1, "max": 512, "step": 1,
                    "tooltip": "像素空间对齐步长，建议 32。随后仍会就近对齐到 VAE 的 16 倍数，因此任意步长不保证最终仍整除 align。",
                }),
                "enable_temporal_chunking": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "近似模式：长视频按时间分块，以复制边界和重叠融合降低激活显存。GroupNorm/注意力跨时间计算，分块结果可能不同。关闭可获得整段计算基准，显存开销更大。",
                }),
                "chunk_size": ("INT", {
                    "default": 32, "min": 1, "max": 4096, "step": 1,
                    "tooltip": "每块的核心潜帧数（非解码后视频帧数）。T 不超过此值时直接整段计算；没有空间分块。",
                }),
                "overlap": ("INT", {
                    "default": 5, "min": 0, "max": 512, "step": 1,
                    "tooltip": "每侧的融合潜帧数，并额外读取同样长度的上下文；每块最多处理 chunk_size + 4×overlap 潜帧。增大不能保证与整段结果一致。",
                }),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = "CK/Open-H3-Regenerate"
    DESCRIPTION = "OpenH3 视频潜空间放大；分块为近似模式。输出位于 CPU，保持输入 dtype 和其他元数据；放大时移除原 noise_mask。"

    def upscale(self, upscaler, latent, width, height, align=32,
                enable_temporal_chunking=True, chunk_size=32, overlap=5):
        from .open_h3.upscaler import upscale_latent
        return (upscale_latent(
            upscaler, latent, width, height, align,
            enable_temporal_chunking, chunk_size, overlap,
        ),)
