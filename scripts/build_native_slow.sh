#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/build_native_slow.sh <isolated-root> [build-threads]" >&2
  exit 2
fi
ROOT="$1"
THREADS="${2:-6}"
REVISION=372dc5ca157276beb6c67b2e92477e8d662e964a
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "Native Metal profile requires macOS arm64; use Docker CPU elsewhere." >&2
  exit 1
fi
if [[ ! "$THREADS" =~ ^[0-9]+$ ]] || (( THREADS < 1 || THREADS > 12 )); then
  echo "Build threads must be in 1..12." >&2
  exit 2
fi
mkdir -p "$ROOT/runtime/tmp" "$ROOT/cache"
ROOT="$(cd "$ROOT" && pwd -P)"
export TMPDIR="$ROOT/runtime/tmp/"
export XDG_CACHE_HOME="$ROOT/cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  python3.13 -m venv --copies "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --no-cache-dir -r requirements/qualification-macos.lock
SOURCE="$ROOT/runtime/llama.cpp"
if [[ ! -d "$SOURCE/.git" ]]; then
  git init "$SOURCE"
  git -C "$SOURCE" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [[ "$(git -C "$SOURCE" remote get-url origin)" != https://github.com/ggml-org/llama.cpp.git ]]; then
  echo "Refusing an unexpected existing runtime checkout." >&2
  exit 1
fi
if git -C "$SOURCE" rev-parse --verify HEAD >/dev/null 2>&1; then
  if [[ "$(git -C "$SOURCE" rev-parse HEAD)" != "$REVISION" || -n "$(git -C "$SOURCE" status --porcelain)" ]]; then
    echo "Existing runtime checkout differs or has changes; no reset/cleanup is performed." >&2
    exit 1
  fi
else
  git -C "$SOURCE" fetch --depth 1 origin "$REVISION"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
CMAKE="$ROOT/.venv/bin/cmake"
"$CMAKE" -S "$SOURCE" -B "$ROOT/runtime/build-metal" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DMTMD_VIDEO=OFF
"$CMAKE" --build "$ROOT/runtime/build-metal" --target llama-server -j "$THREADS"
printf '%s\n' "$REVISION" > "$ROOT/runtime/build-metal/bin/SOURCE_REVISION"
cp "$SOURCE/LICENSE" "$ROOT/runtime/build-metal/bin/LLAMA-LICENSE"
"$ROOT/runtime/build-metal/bin/llama-server" --version
"$ROOT/runtime/build-metal/bin/llama-server" --list-devices
