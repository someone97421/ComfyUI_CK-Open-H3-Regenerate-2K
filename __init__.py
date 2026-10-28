from .nodes_refiner import (
    OpenH3ApplyRegenerate, OpenH3RefineContext, OpenH3SigmaSchedule, OpenH3RefineSampler,
)
from .nodes_media import OpenH3VideoLatent, OpenH3EncodeSilence, OpenH3PrepareVideo, OpenH3Decode
from .nodes_upscale import OpenH3UpscalerLoader, OpenH3LatentUpscale

NODE_CLASS_MAPPINGS = {
    "CKOpenH3ApplyRegenerate": OpenH3ApplyRegenerate,
    "CKOpenH3RefineContext": OpenH3RefineContext,
    "CKOpenH3SigmaSchedule": OpenH3SigmaSchedule,
    "CKOpenH3RefineSampler": OpenH3RefineSampler,
    "CKOpenH3VideoLatent": OpenH3VideoLatent,
    "CKOpenH3EncodeSilence": OpenH3EncodeSilence,
    "CKOpenH3PrepareVideo": OpenH3PrepareVideo,
    "CKOpenH3Decode": OpenH3Decode,
    "CKOpenH3UpscalerLoader": OpenH3UpscalerLoader,
    "CKOpenH3LatentUpscale": OpenH3LatentUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKOpenH3ApplyRegenerate": "OpenH3 Apply Regenerate · 精炼补丁",
    "CKOpenH3RefineContext": "OpenH3 Refine Context · 固定音频条件",
    "CKOpenH3SigmaSchedule": "OpenH3 Sigma Schedule · 噪声调度",
    "CKOpenH3RefineSampler": "OpenH3 Refine Sampler · 独立采样",
    "CKOpenH3VideoLatent": "OpenH3 Video Latent · 提取视频流",
    "CKOpenH3EncodeSilence": "OpenH3 Encode Silence · 静音编码",
    "CKOpenH3PrepareVideo": "OpenH3 Prepare Video · 帧准备",
    "CKOpenH3Decode": "OpenH3 Decode · 原生 VAE 解码",
    "CKOpenH3UpscalerLoader": "OpenH3 Upscaler Loader · 放大模型",
    "CKOpenH3LatentUpscale": "OpenH3 Latent Upscale · 分块放大",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
