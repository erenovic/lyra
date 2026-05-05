FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /workspace

# Target GH200 (sm_90) only — avoids building for all CUDA archs
# TORCH_CUDA_ARCH_LIST: respected by transformer_engine and most CUDA-extension builders
# FLASH_ATTN_CUDA_ARCHS: respected by flash-attn (ignores TORCH_CUDA_ARCH_LIST)
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV FORCE_CUDA=1
ENV FLASH_ATTN_CUDA_ARCHS="90"
ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"
ENV FLASH_ATTENTION_FORCE_CXX11_ABI="TRUE"
# Threads per nvcc invocation
ENV NVCC_THREADS=4
# Compiler processes per package (nvcc/gcc)
ENV MAX_JOBS=16

# System packages:
#   ffmpeg / hdf5-tools                                     — data preprocessing
#   nodejs                                                  — yt-dlp-ejs YouTube signature decryption
#   libeigen3-dev                                           — required by vipe (USE_SYSTEM_EIGEN=1)
#   cmake + libav*-dev / libswscale-dev / libswresample-dev — building decord from source
#   libx11-dev / libgl-dev / libegl-dev                     — building glcontext (moderngl, pulled in by MoGe)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg nodejs rsync hdf5-tools libeigen3-dev \
        cmake build-essential \
        libavformat-dev libavcodec-dev libavfilter-dev \
        libavutil-dev libswresample-dev libswscale-dev \
        libx11-dev libgl-dev libegl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build decord from source. PyPI has no aarch64+cp312 wheel for decord==0.6.0
# (last release is from 2022). CPU-only build is sufficient for dataset loading.
# Using zhanwenchen/decord fork — upstream dmlc/decord is unmaintained and fails
# to build against FFmpeg 6/7 headers in the NGC base image (missing AVBSFContext,
# const AVCodec** signature changes, etc.).
RUN git clone --recursive https://github.com/zhanwenchen/decord /tmp/decord \
    && cd /tmp/decord && mkdir build && cd build \
    && cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) \
    && cd ../python && pip install --no-deps . \
    && rm -rf /tmp/decord

# Sanity check: confirm NGC base image already ships flash-attn for sm_90.
# Fails loud if not — in which case we'd need to build flash-attn explicitly.
RUN python -c "import flash_attn; print(f'flash_attn preinstalled: {flash_attn.__version__}')"

# Copy the project. Submodules (vipe, depth_anything_3) MUST be initialized
# on the host before `podman build` — the editable installs below depend on them:
#   git submodule update --init --recursive
COPY . .

# Python deps from requirements.txt — --no-deps so we don't clobber NGC's
# torch/CUDA stack. Mirrors step 5 of INSTALL.md.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -r requirements.txt

# MoGe needs its own dependency resolution (no --no-deps).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "git+https://github.com/microsoft/MoGe.git"

# transformer_engine — built against NGC's torch.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation "transformer_engine[pytorch]"

# Symlink cuda_runtime as cudart so transformer_engine can find it.
# Tolerated to fail if NGC's torch wheel layout doesn't include nvidia/cuda_runtime.
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && if [ -d "$SITE/nvidia/cuda_runtime" ]; then \
           ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"; \
       fi

# Flash-Attention: NGC PyTorch 25.06 already ships flash-attn prebuilt for sm_90.
# Building 2.6.3 from source fails against this image's libcudacxx (CUDA 12.9).

# Vendored CUDA extensions (editable installs — paths must persist in the image).
RUN --mount=type=cache,target=/root/.cache/pip \
    USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'
