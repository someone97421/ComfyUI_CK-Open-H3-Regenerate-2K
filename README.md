# ComfyUI CK Open-H3-Regenerate-2K

把 Open-H3 的 latent 放大、Refiner 补丁、固定音频条件、噪声调度、采样和 VAE 解码拆成可组合的 ComfyUI 节点。分类为 `CK/Open-H3-Regenerate`，节点 ID 使用 `CKOpenH3` 前缀。

第一版针对本地原生 MiniMax H3 实现适配。复用传入的 `MODEL`、`CONDITIONING`、`VAE` 和 `LATENT`，不加载独立 Diffusers pipeline，不下载模型、不安装依赖、不修改宿主源码。

可导入案例：[官方 H3 R2V → Open-H3-Regenerate 2K](example_workflows/OpenH3_R2V_Regenerate_2K.json)。[案例说明与完整模型下载链接](example_workflows/README.md)包含本机已存在的组件、两个新增精炼权重及接线说明。案例保留草稿与2K两个输出，并复用原始音轨。

## 安装与模型

把本目录放在 `ComfyUI/custom_nodes`，重启 ComfyUI 后搜索 `OpenH3`。无需运行安装命令。本版使用 ComfyUI 环境已有的 PyTorch、safetensors 及原生模块。

沿用现有 H3 模型、Qwen3-VL H3 文本编码器、H3 视频 VAE、H3 音频 VAE。精炼新增的两个文件可放在：

```text
ComfyUI/models/loras/Open-H3-Regenerate-2K/
  open_h3_regenerate_2k_lora.safetensors
  sharp_curve_branches.safetensors
```

节点也接受这两个文件的绝对路径，不要求上述目录。文件来自用户指定的 `Sensen02/Open-H3-Regenerate-2K` 模型仓库。发布版的线性 LoRA 已经合并 Refiner、LightX2V Turbo、LMS Sharp，默认强度为 1；不要为复现默认配方再重复叠加同一组 Turbo/Sharp。

`curve_table=auto` 优先检查传入模型的曲线表，然后只读取本地 H3 REF2VA safetensors 中的小型 `adaln_t_table`。需要 1025×8 FP32 表，参考 SHA256 为：

```text
c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7
```

本机已有的 `minimax_h3_ref2va_pruned_bf16.safetensors` 包含匹配的表；不会把整个文件当成第二个模型加载。也可指定含表的 safetensors 或 `adaln_t_table.bin`。显式指定不同哈希的表会提示日志但仍允许使用。

放大权重继续使用已有的 `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`，无须重复下载。

## 最小接线：接现有 H3 工作流

```text
现有 H3 MODEL ── Apply Regenerate ── Refine Context ────┐
现有 H3 文本 CONDITIONING ────────── Refine Context ────┤
H3 音频 VAE + 视频 LATENT ── Encode Silence ──┘         │
                                                       ├─ Refine Sampler ── Decode ── IMAGE
现有视频 LATENT ── 现有 H3 Latent Upscaler ──────────────┤
Sigma Schedule ─────────────────────────────────────────┘
```

1. 原生 H3 模型加载器的 `MODEL` 接 **Apply Regenerate**。
2. 正常使用现有 H3 文本编码节点，将 `CONDITIONING` 接 **Refine Context**。
3. **Encode Silence** 接 H3 **音频** VAE 和视频 latent，输出接 **Refine Context** 的 `audio_latent`。它会按视频 latent 的长度编码静音，不用零 latent 冒充静音编码。
4. **Refine Context** 输出 `MODEL` 与 `CONDITIONING`，接 **Refine Sampler**。
5. 将放大后的视频 latent 接 **Refine Sampler**。如原流程输出音视频组合 latent，先经 **Video Latent** 提取视频，再送现有放大器。
6. **Sigma Schedule** 选择 `reference_3`，接 `sigmas`。Sampler 默认 CFG=1、原生 Euler。
7. 精炼输出接已有 VAE Decode，或本插件 **Decode**；`IMAGE` 再接已有视频保存节点。FPS、音轨与编码格式继续由视频节点管理。

也可以省略 **Refine Context**，把静音 latent 直接接专用 Sampler 的可选 `audio_latent`。这种用法保留输入 CONDITIONING 中的参考图/首尾帧信息；若要按参考流程清除这些条件，请使用 Context 节点。

## 接已有视频

```text
视频加载 → IMAGE → Prepare Video → 原生 VAE Encode（H3 视频 VAE）
                                   ↓
                              H3 视频 LATENT → 放大 → 精炼 → 解码
```

Prepare Video 默认：1344×768、24fps、121 帧、复制 3 个尾帧，输出 124 帧。非 H3 视频可用 1024×576 作为起点。输出的 `output_frame_count` 是去掉尾部填充后的帧数，可接 Decode 的 `frame_count`。

也可完全使用已有的视频预处理节点。原生 H3 VAE Encode 已处理像素和 latent 归一化，勿额外套一遍 ImageNet 或 latent mean/std。

视频预处理采用 PyTorch bilinear 和按时间索引取帧；它与上游 FFmpeg 的重采样/舍入不保证逐像素相同。原生 VAE 编码使用宿主提供的后验路径，不复刻 NVIDIA 的 FP32 posterior.sample(seed)；现有 H3 latent 输入可绕过这项差异。

## 节点接口

| 节点 | 输入 | 输出 |
|---|---|---|
| Apply Regenerate | MODEL、合并 LoRA 路径、Sharp 路径、曲线表、两个强度 | MODEL、加载报告 STRING |
| Refine Context | MODEL、CONDITIONING、固定音频 LATENT、是否清除参考条件 | MODEL、CONDITIONING |
| Sigma Schedule | reference_3 / reference_2 / shifted / custom | SIGMAS |
| Refine Sampler | MODEL、CONDITIONING、LATENT、SIGMAS、seed、CFG | 视频 LATENT |
| Upscaler Loader | 放大权重路径、精度 | OPEN_H3_UPSCALER |
| Latent Upscale | OPEN_H3_UPSCALER、视频 LATENT、宽高、时间分块参数 | 视频 LATENT |
| Video Latent | 视频或音视频组合 LATENT | 视频 LATENT |
| Encode Silence | H3 音频 VAE、视频 LATENT、FPS | 音频 LATENT |
| Prepare Video | IMAGE、尺寸、输入/输出 FPS、帧数、补帧、裁切方式 | IMAGE、有效帧数、FPS |
| Decode | VAE、LATENT、native/tiled、帧数、中心裁切、tile 参数 | IMAGE |

标准视频 latent 为 `[1,24,T,H,W]`；H3 原生模型的当前前向支持 batch=1。H3 音频 latent 为 `[1,32,2,T_audio]`。放大器可处理普通 `[B,24,T,H,W]`，不负责拆分组合 latent。

Sampler 可选输入：`negative`、`NOISE`、`SAMPLER`、`audio_latent`。不接 SAMPLER 时使用原生 Euler；CFG 不为 1 时连接负面条件。外接 NOISE 时使用该对象生成的噪声及其 seed。内置噪声用私有 RNG，按 H3 patch 顺序生成；`reference_device` 在计算设备上生成，`cpu` 在 CPU 生成，两者同 seed 不保证相同。

只做同分辨率精炼时可绕过放大节点。只做 latent 放大时也不需要加载 Refiner 权重。

## 使用原生高级采样器

**Apply Regenerate → Refine Context** 输出的是标准 `MODEL`，可以接原生 BasicGuider/CFGGuider 和 SamplerCustomAdvanced。将放大后的 latent、Sigma Schedule、Euler KSamplerSelect 和标准 NOISE 连接到高级采样链。原生 sampler 输出可能是音视频组合 latent，解码前使用 Video Latent，或让原生 VAE Decode 自动取视频流。

固定音频 wrapper 会在前向时注入已编码音频、将音频时间固定到 1 并返回零音频更新；视频走宿主正常 flow 采样。参考时间是 `1 - sigma`，初始视频状态是 `(1-sigma0)*latent + sigma0*noise`。原生噪声布局和设备 RNG 可能与专用 Sampler 不同，不能仅用相同 seed 判断等价。

外部模型补丁可以与本插件组合；会保留已有 block patch 并继续调用。其他插件如果绕过原生 block 的 AdaLN，或替换整个前向、不传递 transformer_options，Sharp 分支可能无法工作。编译后的前向与临时 hook 的组合未验证，先使用普通 eager 模型。

## 调度与尺寸

- `reference_3`：`[0.909375, 6/7, 2/3, 0]`。
- `reference_2`：从 0.7 开始的 shift-6 两步调度。
- `shifted`：指定 steps、start_sigma、shift。三步 shifted 并不等于 reference_3。
- `custom`：逗号或空白分隔的 sigma；也可不用此节点，直接接现有 SIGMAS。

外接 SIGMAS 只检查一维和有限数值，不强制单调或零终点，可接 SplitSigmas 等分段调度；空序列或单点序列直接返回视频 latent。非零终点的输出缩放遵循宿主 sampler，不自动保证多次重新加噪的分段调用等价于一次连续采样。参考配方采用 [0,1] 范围。

建议计算尺寸：2K 为 2560×1440；1080P 为 1920×1088，解码中心裁成 1920×1080；4K 为 3840×2176，解码裁成 3840×2160。更高放大倍率可串联两次放大，例如先到 2K，再到 4K。节点不强制按分辨率预设限制工作流。

Decode 的 `frame_count=0` 保留全部帧，宽高设为 0 保留完整画布。默认截取 121 帧，长视频需要自行调整。H3 时长不是简单的 latent T×4；37 个 latent 帧对应编码布局中的 124 帧。

## 5090 32GB 与显存

- H3 基础模型由传入的 ComfyUI ModelPatcher 共享。LoRA 使用宿主局部权重补丁，QKV 与 MLP 的映射在内存中进行，不创建新模型文件。
- 允许原生 H3 量化模型参与加载；具体量化格式的权重 patch 由宿主处理。尚未实测每种量化格式，不按文件名设置白名单。
- 合并 LoRA 文件约 12.45GB，Sharp 因子约 1.24GB，CPU 内存也需要留余量。量化模型不意味着这些 FP32 因子免费。
- Sharp 的 A/B 保留在 CPU，以 FP32 计算当前时间行的小型调制结果，再传到对应 GPU block；降低常驻显存，但增加 CPU 运算和传输。
- 专用 Sampler 默认在成功、异常或中断后卸载本次 H3 模型，让 VAE 有可用显存。`unload_after=False` 保留宿主模型缓存。
- 专项放大器由 ModelPatcher 管理装载和卸载，计算结束会回 CPU。它的完整权重需要装入计算设备；暂未实现逐层卸载或空间切块。
- 时间分块可调 `chunk_size` 与 `overlap`，CPU FP32 融合。GroupNorm 和注意力影响范围跨块，属于近似模式；关闭分块可用作整段参考。设置很大的 overlap 可能比整段更耗显存。
- 实际放大后移除原尺寸的 noise_mask，需在新尺寸上重新设置。其他 latent 元数据浅拷贝保留。
- H3 的 native VAE 路径本身有分块 I/O，也可选择原生 Tiled Decode。后者参数直接交由 ComfyUI 处理。
- 未实现 Refiner 时空 attention 切块，未接入 Sol-Engine / Sol-Attn / CUDA Graph / Fast VAE。未承诺 32GB 下任意分辨率、时长都能运行。

## 宿主隔离与源码结构

导入插件和调用 INPUT_TYPES 仅创建节点描述，不导入推理模块、不访问 GPU、不扫描模型目录。模型文件在节点执行时读取；无自动下载、服务、安装脚本、sys.path/sys.modules 注入或全局目录注册。

Apply 与 Context 返回模型副本；不原地修改输入 CONDITIONING。Sharp 在 block 执行期间安装实例 forward_hook，并通过 finally 移除。固定音频前向只使用本次调用的 options/payload 副本。显存加载与权重 patch 使用 ComfyUI 原有模型管理协议。

```text
__init__.py                 节点注册
nodes_refiner.py            精炼/上下文/调度/采样节点
nodes_media.py              帧准备、静音、视频流、解码
nodes_upscale.py            可选独立放大器节点
open_h3/adapters.py         Diffusers → 原生 H3 LoRA 映射
open_h3/patches.py          Sharp 与固定音频实例补丁
open_h3/weights.py          本地路径和小型曲线表读取
open_h3/sampling.py         原生 guider/采样/噪声/卸载
open_h3/schedules.py        sigma 数学
open_h3/upscaler.py         同源 MIT 放大网络及分块
```

## 核对基线与验证状态

静态核对的 ComfyUI 提交：`08cf0284ff45723aa67841c80cabb0e3cb50c9d7`。环境记录：Windows、RTX 5090、Python 3.12.10、PyTorch 2.11.0+cu130。该记录是源码/API 静态适配基线，不代表该组合已经通过实际推理验收。

NVIDIA 参考分支提交：`92f2759f4b7754673a6edf2fab71a4f51568b664`，路径 `models/Open-H3-Regenerate-2K`。权重核对使用 Sensen02 仓库的模型卡、配置与 LFS 元数据；主要文件的远端 SHA256 与 NVIDIA download.py 预期一致，未下载和逐 tensor 校验所有大权重。

2026-10-28 提交前已在本地 ComfyUI Python 环境执行完整测试套件，30/30 通过，包含 `tests.test_host_isolation`，耗时约 2.5 秒。测试覆盖导入隔离、临时 hook 恢复、模型上下文副本和关键数学映射，使用小型 CPU 张量/受控宿主替身；它们不能代替真实模型输出与显存验证。未执行构建或 GPU 推理。复查命令：

```powershell
python -m unittest discover -s tests -t . -v
python -m unittest tests.test_host_isolation -v
```

提交或推送前必须通过完整套件，且必须包含 `tests.test_host_isolation`。

来源、许可与未内置组件见 `NOTICE.md` 和 `licenses/latent-upscaler-MIT.txt`。
