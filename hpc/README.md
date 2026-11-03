# 超算作业目录

## 目录结构

```
hpc/
├── slurm/
│   ├── submit_infer.sbatch    # 推理冒烟（阶段一验收）
│   ├── submit_sft.sbatch      # 办公 SFT
│   ├── submit_dpo.sbatch      # Agent DPO
│   └── submit_tts.sbatch      # TTS 训练
├── pbs/
│   └── submit_infer.pbs       # PBS 对照模板
├── lsf/
│   └── submit_infer.lsf       # LSF 对照模板
├── env/
│   ├── setup_offline.sh       # 离线建环境（超算上第一个跑）
│   └── modules.env            # module 加载清单
└── README.md
```

## ⚠️ 提交前必须改的 4 处

每个 sbatch 里用 ★ 标了出来：

1. **`--partition=gpu`** → 实际分区名
   ```bash
   sinfo        # 看有哪些分区
   ```
2. **`--qos` / `--account`** → 按集群要求（多数集群必填）
3. **`--time`** → 别一上来就要最大时间，排队反而更久。
   推理 2h / SFT 24h / DPO 12h / TTS 48h 是保守值，跑通后再按实际调
4. **`hpc/env/modules.env`** → 版本号按 `module avail` 核对

## 调度系统对照

| 系统 | 提交 | 查看 | 取消 | 模板 |
| :-- | :-- | :-- | :-- | :-- |
| Slurm | `sbatch x.sbatch` | `squeue -u $USER` | `scancel JOBID` | `hpc/slurm/` |
| PBS | `qsub x.pbs` | `qstat -u $USER` | `qdel JOBID` | `hpc/pbs/` |
| LSF | `bsub < x.lsf` | `bjobs` | `bkill JOBID` | `hpc/lsf/` |

主要变量差异：

| 用途 | Slurm | PBS | LSF |
| :-- | :-- | :-- | :-- |
| 作业号 | `$SLURM_JOB_ID` | `$PBS_JOBID` | `$LSB_JOBID` |
| 节点列表 | `$SLURM_NODELIST` | `cat $PBS_NODEFILE` | `$LSB_HOSTS` |
| 提交目录 | `$SLURM_SUBMIT_DIR` | `$PBS_O_WORKDIR` | `$LS_SUBCWD` |
| GPU 申请 | `#SBATCH --gres=gpu:2` | `#PBS -l ngpus=2` | `#BSUB -gpu "num=2"` |

**确认调度系统后，删掉另外两套模板**——留着只会让人搞混。

## 典型流程

```bash
# 1) 首次：建离线环境（只需一次）
bash hpc/env/setup_offline.sh

# 2) 提交推理冒烟
sbatch hpc/slurm/submit_infer.sbatch

# 3) 看日志
tail -f logs/slurm/infer-*.out

# 4) 跑通后再上训练
sbatch hpc/slurm/submit_sft.sbatch
```

## 交互调试

排队太久时，申请交互式节点调试更快：

```bash
# Slurm
srun --gres=gpu:1 --time=01:00:00 --pty bash

# 进去后直接跑，不用等 sbatch 排队
python -m src.inference.cli dry-run
python -m src.training.sft --config configs/train_sft_office.yaml --dry-run
```

**先用 `--dry-run` 验证配置和数据，再烧机时。**

## 训练完必看

```bash
# 防坍缩记录：期望退出步数有没有塌到 1.0 附近
cat outputs/monitor/exit_history.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['alerts'])"
```

没有这个文件 = 模型未暴露 `expected_exit_step`，
说明防坍缩监控没生效，需人工确认循环层确实在起作用。
