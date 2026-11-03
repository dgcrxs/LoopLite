#!/usr/bin/env bash
# ======================================================================
# 超算环境探测：一条命令跑完，把结果贴回来即可确定打包参数
#
# 用法（在登录节点跑，不需要 GPU，几秒钟完成）：
#     bash probe.sh
#     bash probe.sh > probe.txt 2>&1   # 存成文件方便对比
#
# 它回答四个决定打包参数的硬问题：
#   1. Python 版本  → wheel 的 cpXX 标签（猜错整包作废）
#   2. CUDA 驱动    → torch 该从哪个索引拉、能拉到多新
#   3. 调度系统      → 用哪套作业脚本
#   4. glibc 版本   → manylinux 标签是否兼容
# ======================================================================

set +e   # 探测脚本不能因为某条命令失败就中断

line() { printf '\n===== %s =====\n' "$1"; }

line "系统"
uname -a
[ -f /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
echo "glibc: $(ldd --version 2>/dev/null | head -1)"
echo "架构: $(uname -m)"

line "Python（决定 wheel 标签 cpXX，最关键）"
for c in python3 python3.13 python3.12 python3.11 python3.10 python3.9 python; do
    if command -v "$c" >/dev/null 2>&1; then
        v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
        p=$(command -v "$c")
        echo "  $c -> $v  ($p)"
    fi
done
echo
echo "  conda 环境（如有）:"
command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | head -10 || echo "    （无 conda）"

line "GPU 与 CUDA 驱动（决定 torch 索引）"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    echo
    nvidia-smi | grep -iE 'CUDA Version' || true
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo
    echo "  驱动版本: $drv"
    echo "  → 对照表（选 <= 驱动支持上限的索引）:"
    echo "     cu118 / 驱动>=450    cu124 / 驱动>=550"
    echo "     cu128 / 驱动>=570    cu130 / 驱动>=580"
    echo "     拿不准就选 cu124（兼容面最广，torch 2.6.0）"
else
    echo "  ✗ 无 nvidia-smi（登录节点常见，正常）"
    echo "  请在计算节点执行: srun -p <分区> -N1 --gres=gpu:1 nvidia-smi"
fi

line "调度系统（决定用哪套作业脚本）"
if command -v sbatch >/dev/null 2>&1; then
    echo "  ✓ Slurm: $(sbatch --version 2>&1 | head -1) → 用 hpc/slurm/*.sbatch"
    echo "  分区列表:"
    sinfo -o "%20P %10l %10D %10c %10m %25f" 2>/dev/null | head -15
elif command -v qsub >/dev/null 2>&1; then
    echo "  ✓ PBS: $(qsub --version 2>&1 | head -1) → 用 hpc/pbs/*.pbs"
elif command -v bsub >/dev/null 2>&1; then
    echo "  ✓ LSF: $(lsid 2>&1 | head -1) → 用 hpc/lsf/*.lsf"
else
    echo "  ? 未检测到 sbatch/qsub/bsub，请确认调度系统"
fi

line "模块系统（决定 modules.env 内容）"
command -v module >/dev/null 2>&1 && {
    module avail 2>&1 | grep -iE 'cuda|gcc|python' | head -20
} || echo "  （无 module 命令）"

line "磁盘与配额（放权重需要 ~55GB：FP32 35.5 + BF16 17.8）"
df -h . 2>/dev/null | tail -2
for d in "$HOME" "${SCRATCH:-}" "${WORK:-}" /scratch /lustre /vol; do
    [ -n "$d" ] && [ -d "$d" ] && echo "  $d: $(df -h "$d" 2>/dev/null | tail -1)"
done
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | head -5

line "网络（确认是否真的无外网）"
for u in https://pypi.org https://hf-mirror.com https://download.pytorch.org; do
    code=$(curl -sS -o /dev/null -m 6 -w '%{http_code}' "$u" 2>/dev/null)
    echo "  $u -> ${code:-超时/不可达}"
done

line "总结：把下面这行贴回给助手"
py=$(
  for c in python3 python3.12 python3.11 python3.10 python3.9; do
    command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null && break
  done
)
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
sch=unknown
command -v sbatch >/dev/null 2>&1 && sch=slurm
command -v qsub   >/dev/null 2>&1 && sch=pbs
command -v bsub   >/dev/null 2>&1 && sch=lsf
echo "HPC: python=${py:-?} driver=${drv:-?} gpu=${gpu:-?} sched=$sch"
