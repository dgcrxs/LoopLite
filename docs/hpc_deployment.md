# 超算部署避坑指南（A100 40G）

## 一、资源账：40G 够不够

> 下面所有数字都按**实测 config** 算，不按「9B / 32 层」的直觉估。
> 关键差异：这是**混合注意力**模型，只有 1/4 的层需要存 KV cache。

### 先算 KV cache（最容易估错的一项）

```
只有 full_attention 层有 KV cache，线性注意力层只存固定大小的递归状态。
full_attention_interval = 4  →  32 层里只有 8 层是全注意力

单层单 token 的 KV = 2(K,V) × 4(kv_heads) × 256(head_dim) × 2 bytes(bf16)
                  = 4 KB

循环会让 4..27 层里的 6 个全注意力层各跑 2 遍 → 各需 2 份 cache
有效层数 = 8 + 6 = 14  →  每 token 56 KB
```

| 上下文 | KV cache |
| :-- | :-- |
| 32K | ~1.8 GB |
| 64K | ~3.6 GB |
| 128K | ~7.3 GB |
| 256K | ~14.7 GB |

> ⚠️ 别按「56 层 × 全注意力」去估，那会算出 16GB 这种离谱数字。
> `head_dim=256` 但只有 **4 个 KV 头**（GQA group=4），每层 KV 其实很小。

### 推理

| 项目 | 占用 | 说明 |
| :-- | :-- | :-- |
| BF16 权重 | **17.8 GB** | 9.656B 参数（含视觉塔与 FP32 的 mHC 参数） |
| KV cache（32K） | ~1.8 GB | 见上表 |
| 激活 + 视觉编码器峰值 | ~3 GB | 处理图片/视频时更高 |
| **单卡合计** | **~23 GB** | 40G 有充分余量 |

**结论：单卡 40G 跑 32K～64K 上下文不需要 TP。**
`tensor_parallel=2` 只在两种情况下才值得：
- 需要 ≥128K 上下文（KV 涨到 7GB+）
- 追求吞吐而非延迟（tp2 能摊开算力）

### 微调

| 方案 | 显存 | 说明 |
| :-- | :-- | :-- |
| QLoRA（4bit + LoRA） | ~24 GB | 16K 序列下单卡 40G 可行，推荐 |
| LoRA（BF16 基座） | ~34 GB | 单卡勉强，建议 tp2 或减序列长度 |
| 全参微调 | 远超单卡 | 需 8 卡 DeepSpeed ZeRO-3（官方示例规模） |

QLoRA 明细：4bit 基座 ~5GB + LoRA ~0.3G + 16K 序列激活 ~14G（BPTT 循环放大）+ 优化器 ~2G。
**激活是大头，不是权重**——循环架构要沿时间反传 N 遍，激活随 loop_pass 线性增长。

OOM 时的降级顺序：
1. `loop_pass` 2 → 1（训练阶段最有效，直接砍掉一半 BPTT 图）
2. `max_seq_len` 16384 → 8192 → 4096
3. `per_device_train_batch_size` → 1
4. `gradient_accumulation_steps` 相应调大以保持有效 batch

## 二、CUDA 版本：最容易卡住的一步

官方核对环境是 **CUDA 13.0**，但 A100 集群常见 11.8 / 12.x。

torch 版本必须 ≤ 驱动支持的 CUDA 版本，否则：
```
CUDA driver version is insufficient for CUDA runtime version
```

排查：
```bash
nvidia-smi          # 右上角 CUDA Version = 驱动支持上限
```

**装低不装高**。torch 向下兼容，CUDA 12.4 编译的 torch 能在支持 12.6 的驱动上跑，
反过来不行。

## 三、多卡通信

```bash
export NCCL_DEBUG=WARN          # 出问题改成 INFO 看详细
export NCCL_IB_DISABLE=0        # 有 IB 网络就启用
```

多机多卡若卡住不动，通常是 NCCL 选错了网卡：
```bash
export NCCL_SOCKET_IFNAME=ib0   # 按实际网卡名
```

## 四、文件系统

超算的 `$HOME` 通常：配额小（几十 GB）、IO 慢、多节点不共享。

**必须放对位置**：

| 数据 | 位置 |
| :-- | :-- |
| 代码 | `$HOME` 或工作目录（小，无所谓） |
| 模型权重（FP32 35.5GB + BF16 17.8GB） | 高速并行存储（Lustre/GPFS） |
| 训练数据 | 高速并行存储 |
| checkpoints | 高速并行存储 ⚠️ 会不断增大 |

TTS 训练尤其吃 IO（大量小音频文件），放 `$HOME` 会让 GPU 饿死等数据。

## 五、作业脚本改动点

`hpc/slurm/*.sbatch` 里标了 ★ 的地方必须按集群实际改：

1. `--partition=gpu` → 实际分区名（`sinfo` 查看）
2. `--qos`、`--account` → 按集群要求
3. `--time` → 不要一上来就要最大时间，排队反而更久
4. `hpc/env/modules.env` → `module avail` 核对版本号

调度系统若不是 Slurm：`hpc/pbs/`、`hpc/lsf/` 下有对照模板。

## 六、离线环境变量（必设）

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false     # 否则 DataLoader 多进程会告警/卡死
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 缓解显存碎片
```

漏了 `HF_HUB_OFFLINE` 的后果：代码某处触发联网 → **卡住直到超时**，
不是快速失败。这种挂起最难排查。

## 七、端口转发（用 WebUI / API 时）

超算计算节点通常不对公网开放：

```bash
# 本机执行
ssh -L 7860:localhost:7860 user@hpc        # WebUI
ssh -L 8000:localhost:8000 user@hpc        # API
```

然后本机浏览器访问 `http://localhost:7860`。

WebUI 里 `share=False` 是硬编码的 —— 超算上开公网穿透会失败，且不安全。

## 八、trust_remote_code 的安全提醒

基座是自定义架构，加载时会执行 HF 仓库里的 Python 代码。

**上生产前必须人工审一遍权重目录下的 `*.py`**，
看清楚 `modeling_*.py`、`configuration_*.py` 里有没有异常行为。
Apache 2.0 许可 ≠ 代码无风险。

## 九、阶段一验收标准

跑这个：

```bash
sbatch hpc/slurm/submit_infer.sbatch
```

日志里出现模型输出的文字，就算成功。

**没有 UI、没有量化、没有微调，跑通就是胜利。**
第一次跑通之后再做优化，别在第一步就追求完美。
