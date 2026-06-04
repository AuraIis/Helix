FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_VERSION=2.12.0+cu130
ARG TRITON_VERSION=3.7.0
ARG INSTALL_FLASH_ATTN=0
ARG MAX_JOBS=4

ENV VIRTUAL_ENV=/opt/auralis-perf-venv \
    PATH=/opt/auralis-perf-venv/bin:$PATH \
    PIP_CACHE_DIR=/cache/pip \
    TORCH_CUDA_ARCH_LIST=12.0 \
    FORCE_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
        ninja-build \
        cmake \
        pkg-config \
        rsync \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip setuptools wheel packaging ninja cmake

RUN --mount=type=cache,target=/cache/pip \
    python -m pip install --index-url https://download.pytorch.org/whl/cu130 \
        "torch==${TORCH_VERSION}" \
    && python -m pip install "triton==${TRITON_VERSION}"

RUN --mount=type=cache,target=/cache/pip \
    python -m pip install \
        sentencepiece>=0.2.0 \
        transformers>=4.45.0 \
        datasets>=3.0.0 \
        huggingface_hub>=0.25.0 \
        safetensors>=0.4.5 \
        tqdm>=4.66.0 \
        pyyaml>=6.0.2 \
        numpy>=1.26.0 \
        regex>=2024.9.0 \
        psutil \
        wandb>=0.18.0 \
        einops \
        tabulate

RUN --mount=type=cache,target=/cache/pip \
    python -m pip install "causal-conv1d==1.6.2.post1" --no-build-isolation \
    && python -m pip install "mamba-ssm==2.3.2.post1" --no-build-isolation \
    && python -m pip install "flash-linear-attention==0.5.0"

RUN --mount=type=cache,target=/cache/pip \
    if [ "$INSTALL_FLASH_ATTN" = "1" ]; then \
        python -m pip install "flash-attn==2.8.3" --no-build-isolation; \
    else \
        echo "Skipping flash-attn build because INSTALL_FLASH_ATTN=$INSTALL_FLASH_ATTN"; \
    fi

WORKDIR /workspace/v2data
CMD ["bash"]
