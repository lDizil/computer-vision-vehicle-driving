FROM python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    NAVSIM_REVISION=3e8291bfa89ff247231e0227778840cd0a036896 \
    NUPLAN_REVISION=ce3c323af01c0d7ec5672f7832ef53f9c679aab0

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates git libgdal-dev libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/navsim \
    && git -C /opt/navsim remote add origin https://github.com/autonomousvision/navsim.git \
    && git -C /opt/navsim fetch --depth 1 origin "${NAVSIM_REVISION}" \
    && git -C /opt/navsim checkout --detach FETCH_HEAD \
    && git init /opt/nuplan-devkit \
    && git -C /opt/nuplan-devkit remote add origin https://github.com/motional/nuplan-devkit.git \
    && git -C /opt/nuplan-devkit fetch --depth 1 origin "${NUPLAN_REVISION}" \
    && git -C /opt/nuplan-devkit checkout --detach FETCH_HEAD

# The CPU index supplies +cpu wheels on amd64 and untagged CPU wheels on arm64.
RUN python -m pip install --no-cache-dir pip==25.3 \
    && python -m pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.0.1 torchvision==0.15.2

COPY requirements/ /opt/requirements/

RUN case "$(uname -m)" in \
        aarch64) lock=linux-arm64.lock ;; \
        x86_64) lock=linux-amd64.lock ;; \
        *) echo "Unsupported CPU architecture" >&2; exit 1 ;; \
    esac \
    && python -m pip install --no-cache-dir \
        --build-constraint /opt/requirements/build-constraints.txt \
        -r "/opt/requirements/${lock}" \
    && python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        -e /opt/nuplan-devkit -e /opt/navsim \
    && python -m pip check

ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /workspace
COPY pyproject.toml /workspace/
COPY src/ /workspace/src/
RUN python -m pip install --no-cache-dir --no-build-isolation --no-deps -e . \
    && python -m pip check

RUN useradd --create-home --uid 1000 researcher
USER researcher
COPY --chown=researcher:researcher scripts/ /workspace/scripts/
COPY --chown=researcher:researcher tests/ /workspace/tests/

CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "/workspace/tests"]
