import Config

# Force EXLA to use CPU (host) client for tests.
# Without this, EXLA tries to init a CUDA client which may:
# 1. Fail with CUDA_ERROR_OUT_OF_MEMORY on machines with small/busy GPUs
# 2. Crash the EXLA.Client GenServer
# 3. Cascade to every subsequent test that touches JIT
#
# Run with GPU: CUDA_VISIBLE_DEVICES=0 mix test
config :exla, default_client: :host
