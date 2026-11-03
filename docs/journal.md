# 《LoopLite 开发者指南》· 踩坑日记

按时间倒序。每次踩坑、每次决策都记在这里——
**这些是想不起来就会再踩一遍的东西**。

---

## 2026-11-03 项目启动 · 本机准备完成

### 环境事实

| 项目 | 实测 |
| :-- | :-- |
| 本机 GPU | GTX 950M **2GB** —— 跑不动任何推理，本机定位为纯准备机 |
| 本机 Python | 3.13.14 |
| 包管理 | 无 conda、无 uv，只有 pip 26.1.2 |
| 同步工具 | 有 OpenSSH 10.3，**无 rsync**（所以 sync_hpc.sh 用 scp + tar） |
| 磁盘 | D 盘 921GB 空闲（35.5GB 权重 FP32 + 17.8GB BF16 + ~6GB wheel，绰绰有余） |
| 带宽 | HF 官方不可达；hf-mirror 稳定 ~1.3MB/s（35.5GB 约 7.5 小时） |
| 超算 | A100 40G 多卡，**完全无外网**，scp 直连，调度系统待确认 |

### 决策 1：GGUF / llama.cpp 路线暂不可用

**原方案**把「GGUF Q4_K_M 5.2GB、Q8_0 8.9GB + 打补丁的 llama.cpp」
当作 16G 显卡的核心落地手段。查证后确认：**这条路目前走不通**。

原因（架构层面，不是工程没做）：
1. 基座含**线性注意力**（每 4 层里 3 层），llama.cpp 没有该算子
2. **mHC 双流残差**是 `[B,T,S,D]` 四流结构，llama.cpp 的残差是单流 `[B,T,D]`
3. **循环块**（24 层共享权重跑 2 遍）在 llama.cpp 的计算图里无法表达
   —— 它的图是按顺序层展开的，没有循环边

社区目前也没有该架构的 GGUF 转换脚本或量化版。

**应对**：
- `deployment/patched_llamacpp/` 降级为「观察位」，写了 README 说明未来要做的工作量（约 1–2 人周，主要在线性注意力 CUDA 算子）
- 16G 场景改走 FP8 权重存储 + CPU offload 兜底
- A100 40G 场景走 BF16 + tp2
- 服务化走 LightLLM(tp4) 或 SGLang 指定 PR 分支

### 决策 2：修正「9B」的口径

原方案按 9B 估显存和算力。实际上：

- **存储参数 9.656B**（32 层权重，871 个张量实测累加）
- **前向计算约 14.6B**（中间 24 层跑 2 遍，56 次层计算）

两者必须分开算：**显存按 9.6B，延迟/算力按 14.6B**。

> ⚠️ 这条后面被实测部分推翻，见下方「KV cache 估错了」。

### 决策 3：本机不装 torch

本机 2G 显存，装 torch 没有运行价值。
但完全不装又无法验证核心代码——所以最终方案是：
- `requirements-local.txt`：本机只装下载/打包/静态检查的最小依赖
- 验证时**单独装 CPU 版 torch**，只为跑通 `tests/test_loop_arch.py`

`env_check.py` 里加了 `py_compile` 全量语法检查，
这样即使没装 torch 也能抓出语法错误。

### 决策 4：超算无外网 → 离线打包是本机准备的核心

这一条改变了整个准备工作的重心：
不是「搭个骨架」，而是「把超算需要的一切都在这台 Windows 上备齐」。

最容易踩的坑记在这里：

> **本机 Windows，超算 Linux。**
> 直接 `pip download -r requirements.txt` 会拉到 **Windows 版 wheel**，
> 推上去装不了。`pack_offline.py` 强制指定
> `--platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:`。
>
> 另一个坑：**PyPI 上的 torch 是 CPU 版**，
> 必须走 `https://download.pytorch.org/whl/cu124` 这类官方索引。

`env_check.py` 会扫描 `wheels/` 里有没有带 `win` 字样的 wheel，有就报错。

---

## 2026-11-03 续 · 直接读 checkpoint 把事实钉死

不靠模型卡描述，直接拉 `config.json` + 逐个读 9 个 safetensors 的头部
（HTTP Range 只取前几 MB），把架构与体积算准。**这一步省掉了后面所有的返工。**

### 实测到的事实

| 项目 | 实测值 | 与原先假设的差异 |
| :-- | :-- | :-- |
| 参数量 | **9.656B**（871 张量） | 原先写「约 9B」，偏低了 |
| dtype | **9.413B 是 FP32**，仅 0.243B 是 BF16 | ⚠️ 完全没预料到 |
| 落盘体积 | **35.52 GB** | 原先以为 20GB，**少算了 15GB** |
| 视觉塔 | 0.456B，`qwen3_5_vision` 27 层 | **模型是图文多模态的**，原先当成纯文本 LLM |
| MTP 头 | 0.244B，单独 `mtp.safetensors` | 原先没提 |
| mHC | streams=2, **iterations=20**, fixed_identity=false | 我们代码默认 5，已改成 20 |
| 注意力 | `qwen3_5_text`，3 线性 + 1 全，线性层带 **conv kernel 4** | 官方不是朴素 elu 线性注意力 |

### 坑 1：checkpoint 是 FP32 存的

35.5GB 而不是「9.6B × 2 字节 = 19GB」。

- 磁盘规划直接翻倍，本机留 25GB 会下一半满盘
- 上传量也翻倍 → 所以在**本机先转 BF16**（`src/utils/quantize.py`），
  17.8GB 再传，scp 时间砍一半
- 转换必须逐分片流式（峰值内存 ≈ 单分片 4.7GB），
  整模型 `torch.load` 要 35GB 内存，24G/40G 卡还没开始算就先被内存拖死

### 坑 2：KV cache 估错了，而且是往高估

原日记写「KV 要按 56 层算，按 32 层估会低估 75%」——**反了，而且高估了很多**。

```
实际只有 full_attention 层有 KV cache（线性注意力层只存固定大小递归状态）
full_attention_interval = 4  →  32 层里只有 8 层是全注意力
单层单 token KV = 2 × 4(kv_heads) × 256(head_dim) × 2B = 4KB
循环让 4..27 里的 6 个全注意力层各存 2 份 → 有效 14 层 → 56KB/token

32K 上下文 ≈ 1.8GB（不是 16GB）
```

结论反转：**单卡 40G 跑 32K~64K 根本不需要 tp2**（权重 17.8 + KV 1.8 + 激活 3 ≈ 23GB）。
原方案「必须 tp2」的结论是错的，白白多花一张卡。

### 坑 3：hf-xet 让下载「假死」

`huggingface_hub` 2.x 默认走 Xet 协议，端点在国内容易被墙。
症状很坑：**进程活着、日志不报错、文件大小 30 秒不涨一个字节**。

定位过程：单条 `curl -r 0-52428799` 打同一分片能稳定 1.35MB/s，
说明不是限速也不是文件问题，而是库走的通道不对。

修法：`HF_HUB_DISABLE_XET=1`。已写进 `download_models.py` 默认行为。

### 坑 4：hf-mirror 会拦 Python-urllib 的 UA

自带的 urllib 下载器一开始全是 `403 Forbidden`，
换成浏览器 UA 立刻正常。API 列表接口同理。

### 坑 5：torch 版本不能照抄官方环境

官方验证环境 = torch 2.11 + CUDA 13，但 CUDA 13 要驱动 ≥580，
A100 集群普遍还在 525~570。若照抄，超算上第一步就
`CUDA driver version is insufficient`。

查了实际索引后发现：**transformers 5.18 只要求 torch≥2.5、Python≥3.10**，
所以完全可以在老驱动上跑：

| CUDA | 索引里可用的 torch | 选它 |
| :-- | :-- | :-- |
| cu118 | 最高 2.7.1 | 2.6.0（驱动 ≥450） |
| cu124 | 最高 2.6.0 | 2.6.0（驱动 ≥550）← 默认 |
| cu128 | 2.11.0 | 2.11.0（与官方对齐，驱动 ≥570） |

同时把 `transformers` 钉死 5.18.0（5.22 起要求 Python≥3.11，会挡掉 3.10 的集群）。
`pack_offline.py` 里的 `TORCH_BY_CUDA` 表就是干这个的，并加了单测锁住单调性。

### 坑 6：Windows 上 py_compile 不能写 os.devnull

`env_check.py` 原来用 `cfile=os.devnull` 丢字节码，
Windows 的 devnull 是 `nul`，`py_compile` 直接抛
`FileExistsError: nul is a non-regular file`。改成临时文件。

### 坑 7：线性注意力算错了

`LinearAttention.forward` 里有两处硬伤，是单测抓出来的：

1. GQA 没对齐头数：`k` 是 4 个 KV 头、`q` 是 8 个头，einsum 直接维度报错
2. KV 累加写成了**全局求和** `einsum("bthd,bthe->bhde")`，
   这是非因果的——每个位置都看到未来的 token。改成 `cumsum` 沿时间累和

第 2 点特别隐蔽：不报错、能出结果，只是注意力泄漏了未来信息。
如果不写测试，这个 bug 会一路带到训练里，表现为「loss 异常低但推理很差」。

---

## 待办（下一步要确认/做的事）

- [ ] **在超算上跑 `bash hpc/env/probe.sh`** —— 一条命令同时确认 Python 版本、
      CUDA 驱动、调度系统、glibc，比问人快
- [ ] 按 probe 结果打 wheel（`--python-version X --cuda Y`）。**先探明再打**——
      Python 猜错整包 wheel 作废，CUDA 猜错超算上装不上
- [ ] 权重下载完成后执行 BF16 转换（35.5 → 17.8GB）
- [ ] `python scripts/env_check.py` 无阻塞项后再提交第一个作业

> 💡 **别让两个下载任务并行。** 实测 hf-mirror 单连接 1.35MB/s，
> 并发 4 线程聚合仍是 1.35MB/s（不叠加）。同时下权重和 wheel，
> 模型下载会掉到 0.13MB/s。串行下：权重约 5~7 小时，wheel 约 1 小时。

---

## 风险登记（持续更新）

| 风险 | 现状 | 应对 |
| :-- | :-- | :-- |
| 退出坍缩 | 未发生（还没训） | monitor.py + SmoothExitLoss + 硬设 loop_pass 三级防线 |
| BPTT OOM | 未发生 | 配置已强制 gradient_checkpointing |
| 生态不兼容 | **已发生**（GGUF / vLLM / Ollama 全不支持） | 见决策 1，前期只走原生 transformers |
| Agent 延迟 | 未测 | 默认 loop_pass=1，跑通后做 A/B |
| mHC 参数被量化 | 未发生 | FP32 白名单在 model_base.yaml / quantize.py / 转换器三处保护 |
| 权重下载中断 | **已发生两次**（Xet 假死、403） | 已修：禁 Xet + 带 UA；断点续传可原地重跑 |
| 线性注意力写错 | **已发生并修复** | 单测抓出：GQA 头数未对齐 + KV 漏了因果性 |
| wheel 版本猜错 | 未发生（Python/CUDA 都未确认） | `hpc/env/probe.sh` 先探明；`TORCH_BY_CUDA` 有单测锁单调性 |
