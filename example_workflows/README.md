# 官方 H3 R2V → Open-H3-Regenerate 2K 案例

将同目录 `OpenH3_R2V_Regenerate_2K.json` 拖入 ComfyUI，或从菜单打开工作流。需要重启一次 ComfyUI，让新插件注册节点。

基于 [Comfy-Org 官方 R2V 模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_r2v.json) 改编，保留全部 33 条原有功能连接。案例共 43 个节点（含 5 个说明节点）、52 条连接，仅依赖本插件和本地已有的 ComfyUI 核心节点。

## 导入后做什么

1. 两个 LoadImage 换成你的人物/主体图和场景/风格图。保留的官方图片名只是占位，没有附带图片，也没有读取私人素材。
2. 分别填写 R2V 提示词和精炼提示词。R2V 可用 `<Picture 1>` / `<Picture 2>`；精炼文本直接描述生成画面，不引用这些标签。
3. 下载下方两个精炼权重，或将 Apply Regenerate 的路径改为已有文件位置。
4. 按需改草稿尺寸、时长、目标尺寸、seed 和步数。所有控制都在独立节点中。

默认：1344×768 档草稿、4 步 R2V Turbo、2560×1440 放大、参考 3 步 Euler 精炼、CFG=1。草稿 Turbo 开关关闭后使用基础模型与 20 步普通采样。放大器默认 BF16、时间块 8、重叠 2，为近似低显存起点，不保证峰值。

```text
参考图 + R2V 提示词 → 官方 R2V → 官方采样 → AV latent
                                              ├→ 视频+音频解码 → 草稿输出
                                              └→ 视频流 → latent 放大 → Refiner → 2K解码
                                                                                   │
草稿音频 ──────────────────────────────────────────────────────────────────────────┴→ 2K输出
```

精炼模型直接从共享 UNETLoader 分支，不经过草稿 R2V Turbo。静音编码只作为精炼内部条件，最终保存仍使用 R2V 草稿的音轨。默认 5 秒按官方公式对齐为 124 帧（约 5.17 秒），解码保留全部帧。

## 需要补充的两个模型

放入 `ComfyUI/models/loras/Open-H3-Regenerate-2K/`：

| 文件 | 直链 | 文件大小 |
|---|---|---|
| `open_h3_regenerate_2k_lora.safetensors` | [下载合并 Refiner LoRA](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/main/lora/open_h3_regenerate_2k_lora.safetensors) | 约 12.45 GB |
| `sharp_curve_branches.safetensors` | [下载 Sharp 曲线分支](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/main/lora/sharp_curve_branches.safetensors) | 约 1.24 GB |

使用 NVIDIA `download.py` 指向的 [Efficient-Large-Model 发布仓库](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)。2026-10-28 匿名 HEAD 核对这两个直链均返回 200；仓库公布的 LFS SHA256 分别为：

```text
Refiner: 8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04
Sharp:   26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e
```

与参考源码期望相同。用户原始来源 [Sensen02/Open-H3-Regenerate-2K](https://huggingface.co/Sensen02/Open-H3-Regenerate-2K) 本次匿名请求返回 401，不能确认其当前公开可访问状态；已有同哈希文件可以直接用。本轮没有下载任何大型模型。

## 已存在的本地模型与来源

下列案例路径均已通过文件存在性检查；未重新计算整个大权重的哈希。路径相对 `ComfyUI/models/`。

| 组件 | 案例本地路径 | 来源直链 |
|---|---|---|
| REF2VA 基础模型 | `diffusion_models/Minimax-H3/minimax_h3_ref2va_int8_convrot.safetensors` | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) |
| H3 文本编码器 | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) |
| 视频 VAE | `vae/minimax_h3_video_vae_int8_convrot.safetensors` | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) |
| 音频 VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| 草稿 R2V Turbo | `loras/MinimaxH3/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | [下载](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors) |
| Latent Upscaler | `latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | [下载](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K/resolve/main/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors) |
| 曲线表来源（只读 32KB tensor） | `diffusion_models/Minimax-H3/minimax_h3_ref2va_pruned_bf16.safetensors` | [已有文件来源，不需重复下载](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors) |

上述直链本次匿名 HEAD 均返回 200。加载节点的 `properties.models` 和画布 MarkdownNote 中也保留了模型链接。原生加载器子目录值按 Windows 下拉列表格式填写；其他机器可在下拉框重选。

曲线表也可改为 `auto`，或使用已经提取的 `adaln_t_table.bin`；无需为了此案例再加载一套 REF2VA BF16 模型。

## 可替换的部分

- 将 148/149 两个节点替换为已装的 Minimax H3 3D latent 放大节点：147 的视频输出 → 现有放大器 → 155 的 latent 输入。
- 同分辨率精炼：147 直接接 155 的 latent 输入，绕过放大。
- 用现有 SIGMAS、NOISE 和 SAMPLER 替换对应采样输入。
- 156 可以换成原生 VAE Decode / Tiled Decode。当前默认 native，复用 H3 本身的分块 I/O。
- 草稿与精炼分别保存到 `output/video/OpenH3_R2V_draft_...`、`output/video/OpenH3_R2V_regenerate_2K_...`。如不需要草稿对照，将草稿 SaveVideo 设为 Never；最终音轨分支仍保留。

## 检查范围

完成 JSON 解析、唯一 ID、连接两端与回链、插槽类型、无环图、新增节点参数顺序及输出类型核对；确认官方 33 条功能连接完整保留，模型路径存在性及链接响应已核对。ComfyUI 静态基线为 `08cf0284ff45723aa67841c80cabb0e3cb50c9d7`。

没有打开前端执行导入验收，没有提交队列、加载权重或运行推理。2026-10-28 提交前插件完整测试套件 30/30 通过，包含宿主隔离用例；这不等于案例已通过实际推理验收。模型缺项是上述两个新增权重；参考图片需要用户上传。REF2VA 与量化基础模型是本地复用方案，不宣称与 NVIDIA 原始 Diffusers 模型逐位等价。
