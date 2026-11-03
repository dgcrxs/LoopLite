# ============================================================
# LoopLite 容器镜像
#
# ⚠️ 定位说明：
#   本 Dockerfile 用于「环境一致性」——保证本机、超算、其他机器跑的是同一套依赖。
#   它不能解决 CUDA 驱动兼容问题：容器内仍需与宿主机驱动匹配的 CUDA 版本。
#
#   多数超算用 Singularity/Apptainer 而非 Docker（无 root 权限）。
#   若集群用 Apptainer：
#       apptainer build looplite.sif docker://looplite:latest
#   或直接用离线 wheel 建 venv（hpc/env/setup_offline.sh），更省事。
#
# 构建（在能联网的机器上）：
#   docker build -f deployment/docker/Dockerfile -t looplite:latest .
# ============================================================

# ★ CUDA 版本需与宿主机驱动匹配。A100 常见 11.8 / 12.x
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# 离线环境变量：容器内默认不联网，权重与依赖全部挂载进来
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖分两层：先装 requirements（变动少，利于缓存）
COPY requirements.txt ./
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch 走官方 CUDA 索引（PyPI 上是 CPU 版）
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install -r requirements.txt

# 代码
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY deployment/ ./deployment/
COPY hpc/ ./hpc/

# 权重与数据通过挂载提供（不进镜像，镜像会膨胀到几十 GB）
VOLUME ["/app/models", "/app/data", "/app/logs"]

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "deployment/api_server.py", "--host", "0.0.0.0", "--port", "8000"]
