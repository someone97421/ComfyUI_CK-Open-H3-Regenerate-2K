# 来源与许可记录

## 本插件的实现

节点定义、权重键映射、宿主适配、调度和采样连接代码在本插件内编写，许可见根目录 LICENSE。原生 H3 模型、attention、量化算子、ModelPatcher、guider 和 VAE 直接由已安装 ComfyUI 提供；未复制整套 ComfyUI 或 Diffusers 运行时。ComfyUI 自身的许可证保持不变。

`open_h3/upscaler.py` 中的网络结构、归一化统计和前向计算来自 LBH-123-AI 的 MIT 放大器，原始许可完整保留于 `licenses/latent-upscaler-MIT.txt`；ComfyUI 模型托管、CPU 分块融合及节点拆分在本插件内适配。

## 推理流程参考

- NVIDIA：`https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K`
- 静态核对提交：`92f2759f4b7754673a6edf2fab71a4f51568b664`
- 核对文件：`open_h3/pipeline.py`、`open_h3/adapters.py`、`open_h3/sigmas.py`、`open_h3/upscaler.py`、`tools/merge_adapters.py`、`download.py`。
- 本插件复现其阶段划分、LoRA 布局映射、Sharp 时间曲线、静音音频条件和 flow/Euler 方程，使用 ComfyUI 原生执行机制。
- 本插件没有内置 Sana Pipeline、StageMerger、sys.modules shim、Sol-Engine、Fast VAE、HyperFlow 或其他 NVIDIA 运行时副本。
- 所核对分支的根目录及 Open-H3 子目录未见独立代码许可证文件；不能据此为 NVIDIA 原始源码作出额外授权。本插件 LICENSE 不覆盖该仓库代码或模型权重。

## 放大器

- 项目：`https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`
- 来源：用户本地同名插件的 `nodes/minimax_h3_latent_upscaler_3d.py`。
- 静态 AST 比较确认关键网络类、加载函数及节点执行逻辑与上述 NVIDIA 提交内 `upscaler_node.py` 相同。
- 许可证：MIT，Copyright (c) 2026 LBH-123-AI。
- 权重：`minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`，由用户已有模型路径读取，本插件不分发权重。

## 模型权重

- 用户指定仓库：`https://huggingface.co/Sensen02/Open-H3-Regenerate-2K`
- NVIDIA 当前 download.py 默认发布仓库：`Efficient-Large-Model/Open-H3-Regenerate-2K`。
- 基础模型标记：`MiniMaxAI/MiniMax-H3@6818f6c32d12b210915e44ad56a4228c2608f160`。
- 模型卡声明 Refiner、Fast VAE 和 base_model 遵循 MiniMax H3 Community License；合并的 LightX2V Turbo、LMS Sharp 及放大器仍分别遵循其来源许可。本插件不分发这些权重，也不把本插件代码许可证用于覆盖它们。
- 曲线表源：`Comfy-Org/MiniMax-H3@e5eb578a89295337b8ff433a035929ce0279e0b6` 的 `adaln_t_table`，本插件只从用户本地文件读取。

HF LFS 元数据与 NVIDIA download.py 中的期望值核对如下；此表不是本地大权重实测哈希清单：

| 文件 | SHA256 |
|---|---|
| open_h3_regenerate_2k_lora.safetensors | `8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04` |
| sharp_curve_branches.safetensors | `26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e` |
| minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors | `4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6` |
| fast_h3_vae_noncausal_24l.safetensors（本版未接入） | `8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1` |

大权重、生成媒体、凭据及私人工作流不属于源码交付内容。
