# 🌀 LoopLite（灵环）

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](pyproject.toml)

> 循环生智，轻量无界 · Loop to Infinity, Lite for All

基于 **Looped Transformer（循环架构）** 的私有 AI 引擎。以「计算深度换参数规模」，
打通 **办公自动化、智能体调度、本地 TTS** 三大场景。

**工作模式**：本机（Windows）只做**准备**——搭骨架、下权重、打离线包；
真正的推理与训练全部在**超算（A100 40G 多卡，无外网）**上跑。

---

## 一、当前状态

| 阶段 | 内容 | 状态 |
| :--- | :--- | :--- |
| 准备 | 目录骨架、配置层、核心代码骨架、超算作业脚本、离线打包链路 | ✅ 已就绪 |
| 准备 | 基座权重下载（35.5GB FP32） | ⏳ 进行中 |
| 准备 | wheel 离线包 | ⏳ 进行中 |
| 准备 | FP32 → BF16 转换（35.5→17.8GB，减半上传量） | ⬜ 待执行 |
| 阶段一 | 超算上跑通 Looped 架构推理闭环 | ⬜ 未开始 |
| 阶段二 | 办公 / Agent 数据管线与 LoRA 微调 | ⬜ 未开始 |
| 阶段三 | Flow-matching 轻量 TTS | ⬜ 未开始 |

> 本地 37 个单测全绿（循环层前向、退出坍缩检测、配置、权重转换、打包映射）。

---

## 二、基座模型事实清单（实测核对，非照抄模型卡）

以下数据于 2026-11-03 直接读取 checkpoint 的 `config.json` 与
`safetensors` 头部逐张量累加得出：

| 项目 | 事实 | 备注 |
| :--- | :--- | :--- |
| 模型 | `sensenova/SenseNova-Loop-9B-Preview` | HF + GitHub(`OpenSenseNova/SenseNova-Loop`) |
| 协议 | Apache 2.0 | 可商用 |
| 结构 | 32 层：**首 4 层 + 中间 24 层（共享权重，跑 2 遍）+ 尾 4 层** | 一次前向 = **56 次层计算** |
| 参数 | **9.656B**（871 个张量实测），**前向口径约 14.6B** | ⚠️ 显存按 9.6B 估，算力/延迟按 14.6B 估 |
| **落盘体积** | **35.5 GB** | ⚠️ checkpoint 是 **FP32** 存的，不是 BF16 的 19GB |
| 模态 | **图文多模态**（文本 + 图像 + 视频） | 有视觉塔 0.456B、video_preprocessor_config |
| 注意力 | 混合注意力：每 4 层 = 3 层线性注意力 + 1 层全注意力 | 16 注意力头 / 4 KV 头，head_dim 256 |
| 维度 | hidden 4096，ffn 12288，vocab 248320 | 文本主干为 `qwen3_5_text` |
| 上下文 | 训练最长 **262,144** tokens | 长上下文是它最强的地方 |
| 精度 | checkpoint 整体 FP32；mHC 路由参数须保持 FP32 | BF16 全量转换时白名单保护 `mhc/mixer/sinkhorn` |
| mHC | 2 流、Sinkhorn **20** 次迭代、可学习混合矩阵 | `src/core/loop_transformer.py` 默认值已对齐 |
| 加载 | 必须 `trust_remote_code=True` | 生产部署前需人工审计仓库代码 |
| 版本要求 | Transformers ≥5.18（实测 5.18 只要求 torch≥2.5、Python≥3.10） | 见下方 CUDA 兼容矩阵 |
| 官方部署 | LightLLM（tp4）/ SGLang（需指定 PR 分支） | **vLLM 常规版不支持** |

### CUDA / torch 兼容矩阵（决定离线包怎么打）

官方验证环境是 torch 2.11 + CUDA 13（要驱动 ≥580），但多数 A100 集群还在 525~570。
实测 transformers 5.18 只要求 **torch≥2.5**，所以不必为版本号去换驱动：

| CUDA 索引 | 可用 torch 上限 | 本方案选 | 最低驱动 |
| :--- | :--- | :--- | :--- |
| cu118 | 2.7.1 | 2.6.0 | 450 |
| **cu124（默认）** | 2.6.0 | **2.6.0** | **550** |
| cu128 | 2.11.0 | 2.11.0（与官方对齐） | 570 |
| cu130 | 2.14.1 | 2.14.1 | 580 |

打包时 `--cuda` 决定索引，脚本按 `TORCH_BY_CUDA` 表自动钉死版本。

### ⚠️ 原方案需要修正的三处

1. **GGUF / llama.cpp 路线目前走不通。**
   基座含线性注意力 + mHC 双流混合，llama.cpp 的 GGUF 转换器不认识 `sensenova_loop` 架构，
   社区尚无 Q4_K_M / Q8_0 量化版。`deployment/patched_llamacpp/` 保留为**观察位**，
   不作为阶段一的关键路径。A100 40G 上的替代方案：BF16 + tp2，或 FP8 权重存储。
2. **「9B 跑得动」是存储口径，不是算力口径。**
   中间层跑两遍，56 次层计算的耗时接近 14.6B 稠密模型。延迟预算必须按这个算。
3. **性能对比来自商汤自测**，且 Terminal Bench 2.1、τ³-Banking 两项**落后于** Qwen3.5-27B。
   终端操作类 Agent 与高精度金融规则场景，先做自己的基准测试再决定是否投入。

---

## 三、目录结构

```text
LoopLite/
├── configs/            # 全部超参数集中管理（模型/推理/训练/TTS/集群）
├── data/               # raw → processed（管线脚本在 data/scripts）
├── models/             # base_weights / quantized / loras / tts_weights（不入库）
├── src/
│   ├── core/           # ★ 循环架构实现、稠密转循环配方、权重与显存优化
│   ├── inference/      # 推理引擎（可动态调循环次数）、CLI、WebUI
│   ├── training/       # Trainer 封装、SFT、DPO、★ monitor 防「退出坍缩」
│   ├── apps/           # office（RAG/PPT）/ agent（工具调度）/ tts（flow-matching）
│   └── utils/          # 硬件探测、量化、日志
├── hpc/                # ★ 超算：slurm 作业 + pbs/lsf 对照 + 离线建环境
├── deployment/         # Docker、API Server、llama.cpp 补丁观察位
├── scripts/            # 环境安装、权重下载、★ 离线打包、★ scp 上传、自检
├── tests/              # 循环层前向正确性、退出坍缩检测
└── docs/               # 架构解析、离线准备、超算避坑、Agent 提示词规范
```

---

## 四、本机准备流程

> 前置：Python ≥ 3.11（本机为 3.13），全程不需要 GPU。

### 1. 安装本机最小依赖

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows
pip install -r requirements-local.txt
```

### 2. 拉取基座权重（35.5GB，约 7 小时，支持续传）

```bash
python scripts/download_models.py                      # 自动选引擎，默认 hf-mirror
python scripts/download_models.py --engine native      # 卡住时用这个（裸 HTTP）
```

国内网络两个必修点（脚本已内置）：`HF_HUB_DISABLE_XET=1` 避开被墙的 Xet 端点；
请求带浏览器 UA，否则 hf-mirror 返回 403。

### 3. 本地转 BF16（把上传量减半）

```bash
python -m src.utils.quantize convert \
  --src models/base_weights/SenseNova-Loop-9B-Preview \
  --dst models/quantized/SenseNova-Loop-9B-BF16 \
  --dtype bfloat16
```

35.5GB → 17.8GB，逐分片流式转换，mHC 参数按白名单强制保留 FP32。

**内存注意**：`lm_head` 单张量 248320×4096 FP32 ≈ 4.07GB，整张量读入在 6GB 可用内存的
机器上会 OOM。转换器用 `get_slice()` 沿第 0 维分块（chunk 256MB），
实测峰值约 2.6GB，而不是「源张量 + 目标张量」的 6GB+。

下载还没结束就想挂机转换，用这个（自动等下载完 → 校验 → 转换）：

```bash
python scripts/auto_convert.py --timeout-min 420
```

### 4. 打离线包（代码 + wheel）

```bash
# ⚠️ 先用一条命令探明超算真实环境（Python 版本 / CUDA 驱动 / 调度系统）
bash hpc/env/probe.sh > probe.txt

# 再按探测结果打包（默认 py3.11 + cu124）
python scripts/pack_offline.py --python-version 3.11 --cuda 124
```

Python 版本猜错，整包 wheel 作废；CUDA 索引猜错，超算上装不上 torch。
**探明再打，不要凭感觉。**

### 5. 自检 + 推送

```bash
python scripts/env_check.py        # 有阻塞项会明确列出
bash scripts/sync_hpc.sh --target user@hpc:/path/to/LoopLite
```

---

## 五、超算执行

```bash
# 1) 建环境（完全离线，用本机推上去的 wheel）
bash hpc/env/setup_offline.sh

# 2) 提交作业（Slurm；若为 PBS/LSF 用 hpc/pbs、hpc/lsf 下的对照模板）
sbatch hpc/slurm/submit_infer.sbatch      # 先跑通「输出第一个字」
sbatch hpc/slurm/submit_sft.sbatch        # 办公 LoRA
sbatch hpc/slurm/submit_dpo.sbatch        # Agent 偏好对齐
sbatch hpc/slurm/submit_tts.sbatch        # TTS
```

**阶段一唯一目标**：在超算上跑通 `src/inference/engine.py`，让模型输出第一个字。
没有 UI、没有量化、没有微调——跑通就是胜利。

---

## 六、三大场景策略（A100 40G 适配版）

| 场景 | 部署策略 | 关键调优 |
| :--- | :--- | :--- |
| 办公自动化 | BF16 tp1（≤64K）；≥128K 再上 tp2 | 长文本 RAG 召回 + 结构化输出微调 |
| 智能体 Agent | BF16 tp1，硬限制循环次数（loop_pass=1）控延迟 | DPO 提工具调用准确率，JSON 格式遵守 |
| 本地 TTS | 独立小模型，INT8 | 有声书 vs 实时对话用不同采样步数 |

> 单卡 40G 的实际占用：权重 17.8GB + KV 1.8GB(32K) + 激活 ~3GB ≈ **23GB**。
> 不需要为了「装得下」强行上 tp2——混合注意力只有 1/4 的层存 KV，缓存比想象中小得多。

---

## 七、风险登记

| 风险 | 对策 | 落地位置 |
| :--- | :--- | :--- |
| 退出坍缩（模型跳过循环层偷懒） | 不看 Loss，实时打印每层退出概率分布 + 正则惩罚 | `src/training/monitor.py` |
| BPTT 显存爆炸 | 强制梯度检查点 + QLoRA + 分块 | `src/training/trainer.py` |
| 生态不兼容（vLLM/Ollama 报错） | 前期自写原生推理脚本，不追求一键部署 | `src/inference/engine.py` |
| 超算无外网 | 本机全量离线打包 | `scripts/pack_offline.py` |
| mHC 路由参数被误量化 | 量化白名单保留 FP32 | `src/utils/quantize.py` |

---

## 八、本地验证

不依赖 GPU，纯 CPU 即可跑全部单测：

```bash
python -m pytest tests/ -q          # 55 个用例
python scripts/env_check.py         # 环境与交付物自检
```

`.github/workflows/tests.yml` 已配置 CI（py3.11 / py3.13 双版本 + bash 语法 + YAML 校验）。
⚠️ 当前账号因账单问题被锁，Actions 无法启动作业，推上去的运行会直接失败且日志为空
（`The job was not started because your account is locked due to a billing issue`）。
账单恢复后无需任何改动，推送即自动运行——本地 `pytest` 结果与它一致。

> CI 里有两项容易被忽略的检查值得保留：`bash -n` 扫全部 `.sh`（超算是 Linux，
> 语法错误要在这里暴露而不是提交作业时），以及 `configs/*.yaml` 的可解析性。

---

## 九、许可证

Apache 2.0（与基座一致）。基座权重同样为 Apache 2.0。
