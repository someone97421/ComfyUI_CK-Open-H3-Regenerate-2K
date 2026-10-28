CATEGORY = "CK/Open-H3-Regenerate"


class OpenH3VideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "extract"
    CATEGORY = CATEGORY

    def extract(self, latent):
        from .open_h3.sampling import video_tensor
        result = dict(latent)
        result["samples"] = video_tensor(latent)
        if "noise_mask" in result and result["noise_mask"].is_nested:
            result["noise_mask"] = result["noise_mask"].unbind()[0]
        return (result,)


class OpenH3EncodeSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_vae": ("VAE",),
            "video_latent": ("LATENT",),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "tooltip": "参考为 24fps；依据 H3 潜帧布局推算实际编码时长。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("silence_latent",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(self, audio_vae, video_latent, fps):
        import torch
        from .open_h3.sampling import video_tensor, video_frames
        video = video_tensor(video_latent)
        sample_rate = getattr(audio_vae, "audio_sample_rate", 32000)
        length = round(video_frames(video.shape[2]) / fps * sample_rate)
        waveform = torch.zeros((1, length, 2), dtype=torch.float32)
        encoded = audio_vae.encode(waveform)
        if encoded.ndim != 4 or tuple(encoded.shape[:3]) != (1, 32, 2):
            raise ValueError("请连接 H3 音频 VAE，预期编码结果为 [1,32,2,T]。")
        return ({"samples": encoded.cpu().float(), "sample_rate": sample_rate, "type": "audio"},)


class OpenH3PrepareVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
            "input_fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0}),
            "output_fps": ("FLOAT", {"default": 24.0, "min": 0.1, "max": 240.0}),
            "frame_count": ("INT", {"default": 121, "min": 0, "max": 100000, "tooltip": "0 保留重采样后的完整长度；不足指定长度时复制尾帧。"}),
            "tail_padding": ("INT", {"default": 3, "min": 0, "max": 128, "tooltip": "参考将 121 帧末尾再复制 3 帧，形成 124 帧送入 VAE。"}),
            "fit": (["center_crop", "stretch"],),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("vae_images", "output_frame_count", "fps")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, images, width, height, input_fps, output_fps, frame_count, tail_padding, fit):
        import math
        import torch
        import torch.nn.functional as F
        import comfy.model_management as mm
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[-1] < 3:
            raise ValueError("需要非空 IMAGE 帧序列 [T,H,W,3]。")
        count = frame_count or max(1, round(images.shape[0] * output_fps / input_fps))
        indices = (torch.arange(count, dtype=torch.float64) * input_fps / output_fps).floor().long().clamp(max=images.shape[0] - 1)
        source_h, source_w = images.shape[1:3]
        if fit == "center_crop":
            scale = max(width / source_w, height / source_h)
            size = (max(height, math.ceil(source_h * scale)), max(width, math.ceil(source_w * scale)))
        else:
            size = (height, width)
        output = torch.empty((count + tail_padding, height, width, 3), dtype=torch.float32, device="cpu")
        for start in range(0, count, 4):
            mm.throw_exception_if_processing_interrupted()
            selected = indices[start:start + 4].to(images.device)
            pixels = images.index_select(0, selected)[..., :3].to(device="cpu", dtype=torch.float32).movedim(-1, 1)
            pixels = F.interpolate(pixels, size=size, mode="bilinear", align_corners=False)
            top, left = (size[0] - height) // 2, (size[1] - width) // 2
            output[start:start + len(selected)] = pixels[:, :, top:top + height, left:left + width].movedim(1, -1)
        if tail_padding:
            output[count:] = output[count - 1]
        return output, count, output_fps


class OpenH3Decode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",), "latent": ("LATENT",),
            "mode": (["native", "tiled"], {"tooltip": "直接调用 ComfyUI VAE Decode 或 VAE Decode Tiled；H3 native 路径本身已有分块 I/O。"}),
            "frame_count": ("INT", {"default": 121, "min": 0, "max": 100000, "tooltip": "0 保留所有解码帧。"}),
            "crop_width": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "0 保留宽度；大于 0 时中心裁切，不做缩放。"}),
            "crop_height": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "1080P 可将 1088 行中心裁切为 1080；0 保留高度。"}),
            "tile_size": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 32}),
            "overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 16}),
            "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
            "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 512, "step": 4}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, vae, latent, mode, frame_count, crop_width, crop_height, tile_size, overlap, temporal_size, temporal_overlap):
        import nodes
        if mode == "tiled":
            images, = nodes.VAEDecodeTiled().decode(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)
        else:
            images, = nodes.VAEDecode().decode(vae, latent)
        if frame_count:
            images = images[:frame_count]
        h, w = images.shape[1:3]
        target_h, target_w = crop_height or h, crop_width or w
        if target_h > h or target_w > w:
            raise ValueError(f"裁切尺寸 {target_w}×{target_h} 大于解码尺寸 {w}×{h}。")
        top, left = (h - target_h) // 2, (w - target_w) // 2
        return (images[:, top:top + target_h, left:left + target_w, :],)
