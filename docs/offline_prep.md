# 离线准备手册（本机 → 超算）

> 超算**完全无外网**。所有东西必须在本机准备好再推上去。
> 这份文档是本机工作的全部内容。

## 概览

```
本机 Windows（GTX 950M 2G，只做准备）
    │
    ├─ 1. 建 venv + 装最小依赖        bash scripts/setup.sh
    ├─ 2. 下载基座权重（35.5GB FP32） python scripts/download_models.py
    ├─ 3. 本地转 BF16（35.5→17.8GB）  python -m src.utils.quantize convert ...
    ├─ 4. 下载超算要用的 wheel          python scripts/pack_offline.py --cuda 124
    ├─ 5. 自检                        python scripts/env_check.py
    └─ 6. 推送（只传 17.8GB）         bash scripts/sync_hpc.sh --target user@hpc:/path
                                          │
                                          ▼
超算 Linux（A100 40G ×N，无外网）
    ├─ 7. 建离线环境                   bash hpc/env/setup_offline.sh
    └─ 8. 提交作业                     sbatch hpc/slurm/submit_infer.sbatch
```

> 💡 **为什么在本机转 BF16 而不是上超算再转**：
> 转换是纯 CPU 操作，本机完全能做；而传输量从 35.5GB 降到 17.8GB，
> scp 时间直接砍一半。超算那边只留 FP32 原始件做备份就行。
> 这一步是本次准备里性价比最高的优化。

## 第 1 步：本机环境

```bash
bash scripts/setup.sh
```

只装下载与打包需要的依赖，**不装 torch**（本机 2G 显存跑不动，装了没意义）。
不过如果要执行第 3 步的 BF16 转换，需要 CPU 版 torch：

```bash
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 第 2 步：下载权重

```bash
python scripts/download_models.py
```

**先记住三个数字，不然一定会踩坑：**

| 数字 | 值 | 说明 |
| :-- | :-- | :-- |
| 参数量 | **9.656B** | 实测，871 个张量累加 |
| 落盘体积 | **35.5 GB** | ⚠️ 是 **FP32** 存的，不是 BF16 的 19GB |
| 分片数 | **8** 个 + `mtp.safetensors` | |

- 支持断点续传，中断后**直接重跑**，已完成分片自动跳过
- 默认走 `hf-mirror.com`（`huggingface.co` 国内不通）
- 三个下载引擎，按顺序自动回退：`hf`（官方库）→ `native`（裸 HTTP）→ `modelscope`

### 坑 1：hf-xet 会卡死

`huggingface_hub` 2.x 默认用 Xet 协议，其端点在国内常被墙。
表现是**文件不再增长、进程还在跑**，看起来像卡住。

脚本已默认设 `HF_HUB_DISABLE_XET=1` 强制回落纯 HTTP。
手动排查时也可以直接指定引擎：

```bash
python scripts/download_models.py --engine native --workers 4
```

`native` 引擎是脚本自带的 urllib 实现，只依赖标准库，行为完全可控，
实测在 hf-mirror 上能稳定跑到 ~1.4MB/s（35.5GB 约 7 小时）。

### 坑 2：403 Forbidden

hf-mirror 会拦 `Python-urllib` 的默认 UA。脚本已带浏览器 UA。
自己写脚本拉文件时记得加上，否则拿到的是 403 而不是文件。

### 坑 3：别信模型卡上的体积描述

模型卡没写「FP32 存储」，按 9B×2 字节算会以为只有 19GB，
结果磁盘留 25GB，下到一半满盘。校验以下是硬指标：

```bash
du -sh models/base_weights/SenseNova-Loop-9B-Preview   # 应 ≈ 35.5G
ls models/base_weights/SenseNova-Loop-9B-Preview/model-*-of-00008.safetensors | wc -l  # 应 = 8
```

脚本自动校验：少于 **34GB** 判定不完整。

## 第 3 步：本地转 BF16（强烈建议）

```bash
# 先看计划，不写文件
.venv/Scripts/python.exe -m src.utils.quantize convert \
  --src models/base_weights/SenseNova-Loop-9B-Preview \
  --dst models/quantized/SenseNova-Loop-9B-BF16 \
  --dtype bfloat16 --dry-run

# 真跑（逐分片流式，峰值内存约 7GB）
.venv/Scripts/python.exe -m src.utils.quantize convert \
  --src models/base_weights/SenseNova-Loop-9B-Preview \
  --dst models/quantized/SenseNova-Loop-9B-BF16 \
  --dtype bfloat16
```

产出：35.5GB → **17.8GB**，8 个分片名不变（index.json 原样复用，加载不会出错）。

### 为什么必须保留 mHC 参数为 FP32

转换器按 `configs/model_base.yaml` 的 `keep_fp32_patterns` 白名单
（`mhc` / `mixer` / `sinkhorn` / `hyper_connection`）挑出参数，强制留在 FP32。
把双随机混合矩阵压成 BF16 后，循环几遍信号就会漂，症状是
**长文本后半段突然胡说**——很难归因到量化上，务必别省这一步。

### 可选：再压到 FP8（9.4GB）

```bash
--dtype float8
```

A100 没有 FP8 计算单元，这只能省**存储/传输**，加载后仍需 upcast 回 BF16 计算，
训练场景不建议。推理省显存可以用。

## 第 4 步：打离线包 ⚠️ 最易出错的一步

```bash
python scripts/pack_offline.py --cuda 124 --include-weights
```

### 坑 1：跨平台 wheel

本机是 Windows，超算是 Linux。直接 `pip download -r requirements.txt`
会拉到 **Windows 版 wheel**，推上去装不了。

脚本已强制指定目标平台：
```
--platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:
```

跑完务必确认 `offline_bundle/MANIFEST.json` 里 `wheels.failed` 为空。
`env_check.py` 也会扫一遍有没有 `win` 字样的 wheel——有就说明打错了。

### 坑 2：torch 不能从 PyPI 下

PyPI 上的 `torch` 是 **CPU 版**。必须从 PyTorch 官方索引下 CUDA 版：
```
https://download.pytorch.org/whl/cu124
```

`--cuda` 参数决定索引后缀。A100 集群常见 11.8 / 12.x，先确认：
```bash
# 在超算上执行，看右上角的 CUDA Version
nvidia-smi
```

### 坑 3：这几个包可能下载失败

| 包 | 原因 | 处理 |
| :-- | :-- | :-- |
| `flash-attn` | 需源码编译 | A100 上现场编译（10~30 分钟），或改用 `sdpa`（代码已自动降级） |
| `deepspeed` | 需编译算子 | 不做全参微调就去掉 |
| `bitsandbytes` | 需匹配 CUDA | 失败则改用 torch 原生量化 |
| `chromadb` | RAG 可选 | 不用就去掉 |

失败不会影响打包完成，清单里会单独列出。

## 第 5 步：自检

```bash
python scripts/env_check.py
```

检查 8 类项目：目录结构、关键文件、YAML 合法性、Python 语法、
权重完整性、运行环境、网络通道、离线打包状态。

**提交作业前必跑**。超算上排队几十分钟才发现配置写错，太亏了。

## 第 6 步：推送

```bash
bash scripts/sync_hpc.sh --target user@hpc.example.edu.cn:/vol/home/user/LoopLite
```

- 需要跳板机：在 `~/.ssh/config` 里配 `ProxyJump`，脚本无需改动
- 传的是 BF16 版（17.8GB）；FP32 原始件留在本机备份即可，不必上传
- 中断后重跑即可继续（rsync 续传）
- 只推代码：`--no-weights`；只推权重：`--weights-only`

## 第 7 步：超算建环境

```bash
ssh user@hpc
cd /vol/home/user/LoopLite
bash hpc/env/setup_offline.sh
```

脚本会：
1. 加载 module（`hpc/env/modules.env`，需按集群实际改）
2. 建 venv
3. `pip install --no-index --find-links=wheels -r requirements-hpc.lock.txt`
4. **实测一次 CUDA 计算**（不是只看 `torch.cuda.is_available()`）
5. 校验权重完整性

## 常见失败速查

| 现象 | 原因 | 处理 |
| :-- | :-- | :-- |
| `CUDA driver version is insufficient` | torch 的 CUDA 版本高于驱动支持 | 降 `--cuda` 版本重新打包 |
| `No matching distribution found` | 平台/Python 版本不匹配 | 核对超算 `python3 --version` 后重打包 |
| 权重加载报 key 不匹配 | 权重下载不完整 | 重跑 `download_models.py`，看体积是否 >34GB、分片是否 8 个 |
| 长文本后半段开始胡说 | mHC 参数被压成低精度 | 重跑转换，确认白名单命中（日志里有「保护 FP32 张量 N 个」） |
| pip 卡住不动 | 超算无外网但没设 `--no-index` | 必须带 `--no-index` |
| `trust_remote_code` 报错 | 自定义 modeling 代码没推上去 | 确认 `*.py` 文件随权重目录一起推送 |
