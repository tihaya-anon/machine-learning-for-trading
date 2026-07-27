# ML4T Unified Environment - CPU-only default
# Single Dockerfile builds for both linux/amd64 (x86) and linux/arm64 (Apple Silicon).
#
# Usage:
#   docker compose up ml4t                    # Start Jupyter Lab (CPU)
#   docker compose run --rm ml4t ml4t-status  # Check environment
#   docker compose --profile gpu up ml4t-gpu  # Use the separate GPU image
#
# See envs/README.md for detailed setup instructions.

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

# System tooling only - Python 3.14 comes from uv (python-build-standalone),
# not from the OS. This avoids the deadsnakes PPA's intermittent network
# flakiness on GitHub runners.
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

# Install uv for fast package management.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -s /bin/bash ml4t

# Install Python 3.14 via uv (python-build-standalone) and create venv at
# /opt/ml4t. /opt/ml4t is on PATH later, so /opt/ml4t/bin/python becomes the
# default `python` for downstream RUN steps. No system Python needed.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.14 \
    && uv venv --seed --python 3.14 /opt/ml4t \
    && /opt/ml4t/bin/python --version \
    && /opt/ml4t/bin/pip --version

WORKDIR /build
COPY pyproject.toml uv.lock /build/

# Derive a fully-pinned constraint from uv.lock so the resolve below reproduces
# the locked environment readers get from `uv sync` (single source of truth:
# uv.lock). Without it, `uv pip install` free-resolves to the latest compatible
# release at build time and drifts off the lock.
RUN cd /build && uv export --frozen --no-emit-project --no-hashes --format requirements-txt \
      | grep -iE '^[a-z0-9._-]+==' \
      > /tmp/locked-constraints.txt

# Install the project dependencies. PyTorch resolves from the CPU wheel index via
# pyproject.toml's [tool.uv.sources]; all ordinary PyPI packages use Tsinghua's
# PyPI mirror through PIP_INDEX_URL and uv's command-level --default-index.
RUN /opt/ml4t/bin/pip install --no-cache-dir pip setuptools wheel && \
    uv pip install --python /opt/ml4t/bin/python --no-cache \
    --default-index https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --prerelease=if-necessary-or-explicit \
    --constraint /tmp/locked-constraints.txt /build

# Install optional dependency groups (bayesian, advanced, mlops).
RUN /opt/ml4t/bin/pip install --no-cache-dir \
    "pymc>=5.14" "arviz>=0.18" \
    "opacus>=1.4" \
    "feast>=0.61" "mlflow>=2.16" \
    || echo "WARNING: Some optional deps failed"

# Python 3.14 compatibility: protobuf 4.x uses metaclass tp_new (crashes).
# Force protobuf 5+ which uses the pure-Python descriptor pool.
RUN /opt/ml4t/bin/pip install --no-cache-dir "protobuf>=5.0"

# JupyterLab theme.
RUN /opt/ml4t/bin/pip install --no-cache-dir "catppuccin-jupyterlab"

WORKDIR /app
ENV PATH="/opt/ml4t/bin:$PATH"
ENV PYTHONPATH="/app"

# Jupyter runs without a token so the reader's first command is the only step.
# That is safe only because docker-compose.yml publishes 8888 on 127.0.0.1 alone.
# Origin and XSRF checks stay ON: loopback does not stop a page the reader browses
# from POSTing to localhost:8888, and with no token those checks are the only thing
# standing between a visited web page and code execution on the reader's machine.
ENV JUPYTER_ENABLE_LAB=yes
RUN mkdir -p /etc/jupyter && \
    echo "c.ServerApp.token = ''" >> /etc/jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /etc/jupyter/jupyter_server_config.py

# Disable catboost-widget extension (causes "require is not defined" errors in
# JupyterLab 4.x). The widget is bundled with catboost but is not compatible with
# modern JupyterLab.
RUN mkdir -p /etc/jupyter/labconfig && \
    echo '{"disabledExtensions": {"catboost-widget": true}}' > /etc/jupyter/labconfig/page_config.json

# Test infrastructure - pytest + helpers so CI doesn't need runtime pip install.
RUN /opt/ml4t/bin/pip install --no-cache-dir \
    "pytest>=8.0" "pytest-timeout>=2.3"

COPY envs/test_all_imports.py /opt/ml4t/bin/test_imports.py
RUN chmod +x /opt/ml4t/bin/test_imports.py

RUN echo '#!/bin/bash\n\
echo "============================================"\n\
echo "ML4T Environment Status"\n\
echo "============================================"\n\
python -c "\n\
import torch\n\
print(f\"PyTorch version: {torch.__version__}\")\n\
print(f\"CUDA available: {torch.cuda.is_available()}\")\n\
print(\"Default image: CPU-only PyTorch\")\n\
"\n\
echo ""\n\
echo "Run: ml4t-test-imports     # Test all chapter imports"\n\
echo "Run: ml4t-test-imports 15  # Test specific chapter"\n\
echo "Use: docker compose --profile gpu run --rm ml4t-gpu ...  # GPU image"\n\
echo "============================================"' > /usr/local/bin/ml4t-status && chmod +x /usr/local/bin/ml4t-status

RUN echo '#!/bin/bash\npython /opt/ml4t/bin/test_imports.py "$@"' > /usr/local/bin/ml4t-test-imports && \
    chmod +x /usr/local/bin/ml4t-test-imports

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
