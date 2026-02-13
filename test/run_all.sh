#!/usr/bin/env bash
# Run full Exmc test suite including distributed tests.
#
# Usage:
#   ./test/run_all.sh              # all tests
#   ./test/run_all.sh --only distributed  # distributed only
#   ./test/run_all.sh --trace      # verbose output
#
# Prerequisites:
#   - CUDA GPU with cuDNN 9.x installed
#   - libdevice symlink (created automatically below)

set -euo pipefail
cd "$(dirname "$0")/.."

# Ensure cuda_sdk_lib/nvvm/libdevice/ exists for XLA to find libdevice.10.bc
if [ ! -f cuda_sdk_lib/nvvm/libdevice/libdevice.10.bc ]; then
  echo "Setting up cuda_sdk_lib symlink for XLA libdevice..."
  mkdir -p cuda_sdk_lib/nvvm/libdevice
  ln -sf /usr/lib/nvidia-cuda-toolkit/nvvm/libdevice/libdevice.10.bc \
    cuda_sdk_lib/nvvm/libdevice/libdevice.10.bc
fi

# Run tests (pass through any extra args)
exec mix test "$@"
