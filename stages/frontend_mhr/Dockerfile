# Mesh2Sim frontend_mhr stage — production container.
#
# Build from the monorepo root:
#   docker build -f stages/frontend_mhr/Dockerfile -t mesh2sim-frontend-mhr .
#
# Strict scope: SAM 3D Body inference (vendored under stages/frontend_mhr/third_party/)
# + mesh2sim-contracts. NEVER install OpenSim, nimblephysics, MoGe, vtk, or any
# downstream library here.
#
# Base + pin policy mirrors the prod FastSAM3DToOpenSim Dockerfile so that
# torch / CUDA / cuDNN versions are identical to the validated production stack.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------- #
# System packages                                                              #
# --------------------------------------------------------------------------- #
# gcc + python3-dev are required at runtime because Triton (torch.compile)
# generates and builds CUDA kernel wrappers on the fly.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates unzip \
        ffmpeg \
        libgl1 libglib2.0-0 \
        libgomp1 libegl1 libxrender1 libxext6 \
        libsm6 libx11-6 \
        gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Miniforge → /opt/conda                                                       #
# --------------------------------------------------------------------------- #
RUN wget -q \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"
SHELL ["/bin/bash", "-c"]

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5 \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------------------------- #
# Env mesh2sim-frontend-mhr (Python 3.11)                                       #
# Split into multiple RUN layers so a transient network failure doesn't        #
# wipe out hours of work, and Docker can cache each step independently.        #
# --------------------------------------------------------------------------- #

# Step 1: create env + base build tools
RUN conda create -y -n mesh2sim-frontend-mhr python=3.11 pip && \
    /opt/conda/envs/mesh2sim-frontend-mhr/bin/pip install \
        numpy cython setuptools virtualenv

# Step 2: chumpy from conda-forge (no reliable PyPI wheel on modern Python)
RUN conda install -y -n mesh2sim-frontend-mhr -c conda-forge chumpy && \
    conda clean -afy

# Step 3: PyTorch CUDA 12.4 (big download, isolated layer for caching)
RUN /opt/conda/envs/mesh2sim-frontend-mhr/bin/pip install \
        torch==2.5.1+cu124 \
        torchvision==0.20.1+cu124 \
        --extra-index-url https://download.pytorch.org/whl/cu124

# Step 4: TensorRT (.engine inference path)
RUN /opt/conda/envs/mesh2sim-frontend-mhr/bin/pip install \
        tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs

# --------------------------------------------------------------------------- #
# Application code                                                             #
# --------------------------------------------------------------------------- #
# Copy only what we need from the monorepo build context:
#   - contracts/           : the contracts package (editable)
#   - stages/frontend_mhr/ : the frontend module + vendored sam_3d_body
# Anything else in the monorepo is excluded via .dockerignore.

WORKDIR /workspace
COPY contracts /workspace/contracts
COPY stages/frontend_mhr /workspace/stages/frontend_mhr

# Step 5: contracts in editable mode (light deps only, fast)
RUN /opt/conda/envs/mesh2sim-frontend-mhr/bin/pip install \
        -e /workspace/contracts

# Step 6: frontend module in editable mode (pulls the rest of the deps from
# its pyproject.toml — torch is already installed and won't be touched).
WORKDIR /workspace/stages/frontend_mhr
RUN /opt/conda/envs/mesh2sim-frontend-mhr/bin/pip install -e ".[dev]"

# Final cleanup
RUN conda clean -afy && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------- #
# Runtime configuration                                                        #
# --------------------------------------------------------------------------- #
# Make the conda env's python the default — no `conda activate` boilerplate.
ENV PATH="/opt/conda/envs/mesh2sim-frontend-mhr/bin:$PATH"

# Volumes for mutable / large artefacts:
#   - /workspace/stages/frontend_mhr/checkpoints : downloaded model weights
#   - /workspace/inputs                          : videos to process
#   - /workspace/outputs                         : produced BodyEstimate contracts
VOLUME ["/workspace/stages/frontend_mhr/checkpoints", "/workspace/inputs", "/workspace/outputs"]

CMD ["bash"]
