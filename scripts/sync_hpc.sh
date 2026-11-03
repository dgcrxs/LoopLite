#!/usr/bin/env bash
# ============================================================
# 推送到超算（scp 直连）
#
# 用法：
#   bash scripts/sync_hpc.sh --target user@hpc.example.edu.cn:/vol/home/user/LoopLite
#   bash scripts/sync_hpc.sh --target user@hpc:/path --no-weights   # 只推代码与依赖
#   bash scripts/sync_hpc.sh --target user@hpc:/path --weights-only # 只推权重
#
# 跳板机：在 ~/.ssh/config 里配好 ProxyJump，本脚本无需改动
#   Host hpc
#       HostName 10.x.x.x
#       User yourname
#       ProxyJump bastion
#
# 建议：20GB 权重单独传，避免一次中断前功尽弃
# ============================================================

set -uo pipefail

TARGET=""
NO_WEIGHTS=0
WEIGHTS_ONLY=0
BANDWIDTH=""      # 例: -l 20000 表示限速 20000 Kbit/s

usage() {
    grep '^#' "$0" | sed -n '2,20p' | sed 's/^# //'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --no-weights) NO_WEIGHTS=1; shift ;;
        --weights-only) WEIGHTS_ONLY=1; shift ;;
        --limit) BANDWIDTH="-l $2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "❌ 必须指定 --target"
    usage
fi

cd "$(dirname "$0")/.."
echo "本地目录: $(pwd)"
echo "目标: $TARGET"

# 解析 host 与 path
HOST="${TARGET%%:*}"
RPATH="${TARGET#*:}"
if [ "$HOST" = "$TARGET" ]; then
    echo "❌ 目标格式应为 user@host:/path"
    exit 1
fi

SSH_OPTS="-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60"
SCP_OPTS="$SSH_OPTS $BANDWIDTH"

# ---- 0. 连通性 ----
echo "===== 检查连通性 ====="
if ! ssh $SSH_OPTS -o ConnectTimeout=15 "$HOST" "echo ok" >/dev/null 2>&1; then
    echo "❌ 无法连接 $HOST。检查："
    echo "   - ~/.ssh/config 是否配好（含 ProxyJump）"
    echo "   - 是否需要先登录跳板机/开 VPN"
    exit 1
fi
echo "✓ 连通"

# ---- 1. 远端建目录 ----
echo "===== 准备远端目录 ====="
ssh $SSH_OPTS "$HOST" "mkdir -p '$RPATH' '$RPATH/logs/slurm' '$RPATH/logs/pbs' '$RPATH/logs/lsf'"
echo "✓ 目录就绪"

# ---- 2. 代码 ----
if [ "$WEIGHTS_ONLY" -eq 0 ]; then
    echo "===== 推送代码 ====="
    # 优先推 tar 包（快），没有则逐目录推
    if [ -f dist/looplite-code.tar.gz ]; then
        scp $SCP_OPTS dist/looplite-code.tar.gz "$HOST:$RPATH/"
        ssh $SSH_OPTS "$HOST" "cd '$RPATH' && tar -xzf looplite-code.tar.gz && rm -f looplite-code.tar.gz"
        echo "✓ 代码包已解压"
    else
        echo "  ⚠️ 未找到 dist/looplite-code.tar.gz，改用目录推送"
        for d in src configs scripts hpc docs tests deployment data; do
            [ -d "$d" ] && scp $SCP_OPTS -r "$d" "$HOST:$RPATH/"
        done
    fi
    for f in requirements.txt requirements-hpc.lock.txt pyproject.toml README.md LICENSE .gitignore; do
        [ -f "$f" ] && scp $SCP_OPTS "$f" "$HOST:$RPATH/"
    done
    echo "✓ 代码推送完成"
fi

# ---- 3. wheel 依赖 ----
if [ "$WEIGHTS_ONLY" -eq 0 ] && [ -d wheels ]; then
    echo "===== 推送依赖 wheel ====="
    N=$(ls wheels/*.whl wheels/*/*.whl 2>/dev/null | wc -l)
    echo "  共 $N 个 wheel"
    if [ "$N" -gt 0 ]; then
        # wheel 按 py311-cu124 这样的组合分子目录，按组合分别打包，避免超算上混版
        for d in wheels/*/; do
            [ -d "$d" ] || continue
            name=$(basename "$d")
            lock="requirements-hpc.${name}.lock.txt"
            echo "  → $name"
            if [ -f "$lock" ]; then
                tar -czf "dist/wheels-${name}.tar.gz" "$d" "$lock"
            else
                tar -czf "dist/wheels-${name}.tar.gz" "$d"
            fi
            scp $SCP_OPTS "dist/wheels-${name}.tar.gz" "$HOST:$RPATH/"
            ssh $SSH_OPTS "$HOST" "cd '$RPATH' && tar -xzf wheels-${name}.tar.gz && rm -f wheels-${name}.tar.gz"
        done
        # 兼容旧布局（wheel 直接躺在 wheels/ 下）
        if ls wheels/*.whl >/dev/null 2>&1; then
            tar -czf dist/wheels.tar.gz -C . wheels
            scp $SCP_OPTS dist/wheels.tar.gz "$HOST:$RPATH/"
            ssh $SSH_OPTS "$HOST" "cd '$RPATH' && tar -xzf wheels.tar.gz && rm -f wheels.tar.gz"
        fi
        echo "✓ wheel 推送完成"
    fi
else
    echo "  跳过（无 wheels/ 目录）"
fi

# ---- 4. 权重 ----
if [ "$NO_WEIGHTS" -eq 0 ]; then
    # 优先推 BF16 转换版（17.8GB），没有才推 FP32 原始件（35.5GB）
    if [ -d "models/quantized/SenseNova-Loop-9B-BF16" ] && \
       [ -n "$(ls -A models/quantized/SenseNova-Loop-9B-BF16 2>/dev/null)" ]; then
        W="models/quantized/SenseNova-Loop-9B-BF16"
        SZ="约 17.8GB"
    else
        W="models/base_weights/SenseNova-Loop-9B-Preview"
        SZ="约 35.5GB FP32（未转 BF16，传输量翻倍）"
    fi
    if [ -d "$W" ] && [ -n "$(ls -A "$W" 2>/dev/null)" ]; then
        echo "===== 推送权重（$SZ，耗时较长）====="
        echo "  源: $W"
        echo "  提示：若中断，重跑本脚本即可（scp 会覆盖，已完成的会很快跳过）"
        ssh $SSH_OPTS "$HOST" "mkdir -p '$RPATH/$W'"
        scp $SCP_OPTS "$W"/* "$HOST:$RPATH/$W/" && echo "✓ 权重推送完成"
    else
        echo "  跳过（本地无权重，先跑 python scripts/download_models.py）"
    fi
fi

# ---- 5. 数据 ----
if [ "$WEIGHTS_ONLY" -eq 0 ] && [ -d data/processed ] && [ -n "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "===== 推送处理好的数据 ====="
    scp $SCP_OPTS -r data/processed "$HOST:$RPATH/data/" && echo "✓ 数据推送完成"
fi

echo ""
echo "===== 完成 ====="
echo "下一步（在超算上）："
echo "  ssh $HOST"
echo "  cd $RPATH"
echo "  bash hpc/env/setup_offline.sh"
