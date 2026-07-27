# ML4T GPU Environment - linux/amd64 + NVIDIA only
#
# This image preserves the CUDA PyTorch and CUDA LightGBM path that used to live
# in the default ml4t image. The default envs/ml4t/Dockerfile is CPU-only.
#
# Usage:
#   docker compose --profile gpu up ml4t-gpu
#   docker compose --profile gpu run --rm ml4t-gpu python notebook.py
#
# Requires: NVIDIA GPU + nvidia-container-toolkit

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Use Tsinghua Ubuntu mirrors for apt packages.
RUN sed -i \
      -e 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
      -e 's@http://security.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
      -e 's@https://archive.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
      -e 's@https://security.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' \
      /etc/apt/sources.list.d/ubuntu.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git libhdf5-dev pkg-config wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source.
RUN curl -L https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz | tar xz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf ta-lib-0.6.4

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -s /bin/bash ml4t

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.14 \
    && uv venv --seed --python 3.14 /opt/ml4t \
    && /opt/ml4t/bin/python --version \
    && /opt/ml4t/bin/pip --version

ARG TARGETARCH
RUN if [ "${TARGETARCH:-amd64}" != "amd64" ]; then \
      echo "The ML4T GPU image supports linux/amd64 only." >&2; \
      exit 1; \
    fi

# Install PyTorch FIRST (rarely changes - maximize cache reuse).
# CUDA 12.8 wheels bundle the CUDA runtime and still import on CPU-only machines,
# but this image is intended for NVIDIA runtime use.
RUN /opt/ml4t/bin/pip install --no-cache-dir pip setuptools wheel && \
    /opt/ml4t/bin/pip install --no-cache-dir \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN /opt/ml4t/bin/pip freeze | grep -E '^(torch|torchaudio|torchvision)==' > /tmp/ml4t-constraints.txt

# Pre-install the ml4t libraries so they land in their own cached layer. Exact
# versions are enforced by the uv.lock-derived constraint applied to the resolve
# step below, so this only needs the lower bounds (--pre allows the pre-only builds).
RUN /opt/ml4t/bin/pip install --no-cache-dir --pre \
    "ml4t-data>=0.1.0b15" \
    "ml4t-diagnostic>=0.1.0b25" \
    "ml4t-engineer>=0.1.0b8" \
    "ml4t-live>=0.1.0b3" \
    "ml4t-backtest>=0.1.0b16" \
    "ml4t-models>=0.1.0a1"

# Install CUDA toolkit (nvcc + headers) for building LightGBM with CUDA.
# Minimal install: nvcc-12.6 + runtime dev headers. CUDA is backward-compatible:
# 12.6 nvcc compiles for the CUDA runtime bundled with the PyTorch wheel.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-nvcc-12-6 cuda-cudart-dev-12-6 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="${PATH:+${PATH}:}/usr/local/cuda-12.6/bin"

WORKDIR /build
COPY pyproject.toml uv.lock /build/

# Derive a fully-pinned constraint from uv.lock so the resolve below reproduces
# the locked environment readers get from `uv sync`. The torch stack is excluded
# because this image intentionally preinstalls CUDA wheels from the PyTorch CUDA
# index. lightgbm is excluded because it is built with CUDA below.
RUN cd /build && uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
      | grep -iE '^[a-z0-9._-]+==' \
      | grep -viE '^(torch|torchvision|torchaudio|lightgbm|triton|setuptools)==|^nvidia-' \
      > /tmp/locked-constraints.txt

# Remove packages and source overrides that need special handling from this
# temporary pyproject copy:
# - torch: already installed from CUDA wheels above
# - lightgbm: installed separately with CUDA
# - PyTorch CPU-only source/index: belongs to the default CPU image/local sync
RUN sed -i \
      -e '/# BEGIN: PyTorch CPU-only default source/,/# END: PyTorch CPU-only default source/d' \
      -e '/# BEGIN: PyTorch CPU-only index/,/# END: PyTorch CPU-only index/d' \
      -e '/"lightgbm/d' \
      -e '/"torch>=/d' \
      /build/pyproject.toml

# Install remaining dependencies without replacing the preinstalled CUDA torch
# stack. --constraint pins every package to its uv.lock version (reproducible,
# no drift). --override keeps the preinstalled torch stack.
RUN echo 'cvxpy>=1.5' >> /tmp/ml4t-constraints.txt && \
    uv pip install --python /opt/ml4t/bin/python --no-cache \
    --default-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --prerelease=if-necessary-or-explicit \
    --constraint /tmp/locked-constraints.txt \
    --override /tmp/ml4t-constraints.txt /build

# LightGBM CUDA build from source. Build fails if CUDA setup is broken.
# Full CUDA verification (lgb.train device=cuda) requires a GPU at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libnccl-dev libnccl2 gcc-12 g++-12 && \
    rm -rf /var/lib/apt/lists/* && \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-12 \
    /opt/ml4t/bin/pip install --no-cache-dir --force-reinstall --no-deps \
      --no-binary lightgbm \
      --config-settings=cmake.define.USE_CUDA=ON "lightgbm>=4.6" && \
    /opt/ml4t/bin/python -c "\
import lightgbm as lgb; \
print(f'LightGBM {lgb.__version__}'); \
print('CUDA build: OK (runtime verification requires GPU)')"

RUN /opt/ml4t/bin/pip install --no-cache-dir \
    "pymc>=5.14" "arviz>=0.18" \
    "opacus>=1.4" \
    "feast>=0.61" "mlflow>=2.16" \
    || echo "WARNING: Some optional deps failed"

RUN /opt/ml4t/bin/pip install --no-cache-dir "protobuf>=5.0"

WORKDIR /app
ENV PATH="/opt/ml4t/bin:$PATH"
ENV PYTHONPATH="/app"

# nvidia-container-toolkit passes through /dev/nvidia* and sets NVIDIA_* envs.
ENV JUPYTER_ENABLE_LAB=yes
RUN mkdir -p /etc/jupyter && \
    echo "c.ServerApp.token = ''" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /etc/jupyter/jupyter_server_config.py

RUN mkdir -p /etc/jupyter/labconfig && \
    echo '{"disabledExtensions": {"catboost-widget": true}}' > /etc/jupyter/labconfig/page_config.json

RUN /opt/ml4t/bin/pip install --no-cache-dir \
    "pytest>=8.0" "pytest-timeout>=2.3"

COPY envs/test_all_imports.py /opt/ml4t/bin/test_imports.py
RUN chmod +x /opt/ml4t/bin/test_imports.py

RUN echo '#!/bin/bash\n\
echo "============================================"\n\
echo "ML4T GPU Environment Status"\n\
echo "============================================"\n\
python -c "\n\
import torch\n\
print(f\"PyTorch version: {torch.__version__}\")\n\
print(f\"CUDA available: {torch.cuda.is_available()}\")\n\
if torch.cuda.is_available():\n\
    print(f\"CUDA version: {torch.version.cuda}\")\n\
    print(f\"GPU count: {torch.cuda.device_count()}\")\n\
    for i in range(torch.cuda.device_count()):\n\
        print(f\"  GPU {i}: {torch.cuda.get_device_name(i)}\")\n\
else:\n\
    print(\"CUDA runtime installed, but no GPU is visible to this container\")\n\
"\n\
echo ""\n\
echo "Run: ml4t-test-imports     # Test all chapter imports"\n\
echo "Run: ml4t-test-imports 15  # Test specific chapter"\n\
echo "============================================"' > /usr/local/bin/ml4t-status && chmod +x /usr/local/bin/ml4t-status

RUN echo '#!/bin/bash\npython /opt/ml4t/bin/test_imports.py "$@"' > /usr/local/bin/ml4t-test-imports && \
    chmod +x /usr/local/bin/ml4t-test-imports

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
