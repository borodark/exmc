defmodule Exmc.JIT do
  @moduledoc """
  Runtime JIT backend abstraction.

  Auto-detects available compilers in priority order: EXLA > EMLX > Evaluator.

  - **EXLA**: CUDA/ROCm/CPU acceleration, f64 supported. Default on Linux.
  - **EMLX**: Metal GPU acceleration on macOS via MLX, f32 only. Default on macOS
    when EXLA is not installed.
  - **Vulkan**: Cross-platform GPU compute (FreeBSD NVIDIA, Linux NVIDIA/AMD/Intel,
    macOS via MoltenVK). f32 compute, f64 storage. No kernel fusion in v0.1.
    Opt-in via `config :exmc, :compiler, :vulkan`.
  - **Evaluator**: Pure Elixir fallback (BinaryBackend). Very slow but always works.

  ## Configuration

  Override auto-detection via application config:

      config :exmc, :compiler, :emlx     # force EMLX
      config :exmc, :compiler, :exla     # force EXLA
      config :exmc, :compiler, :vulkan   # force Vulkan (FreeBSD GPU path)
      config :exmc, :compiler, :none     # disable JIT

  ## EMLX Precision

  EMLX (MLX) operates in f32 only — Metal GPU has no f64 support.
  When EMLX is active, all model tensors are automatically downcast to f32.
  This is sufficient for most models but may cause numerical issues with
  very steep log-density gradients or long chains.
  """

  @doc """
  JIT-compile a function using the best available compiler.

  Accepts the same opts as `EXLA.jit/2`. When EMLX is the active compiler,
  EXLA-specific options (like `client:`) are translated to EMLX equivalents.
  When no compiler is available, returns the function unchanged (Evaluator path).
  """
  def jit(fun, opts \\ []) do
    case detect_compiler() do
      nil ->
        fun

      Nx.Vulkan ->
        # Vulkan has no Nx.Defn.Compiler of its own (no kernel fusion in
        # v0.1). Dispatch each defn op through Nx.Defn.Evaluator with
        # Nx.Vulkan.Backend as the default — every Nx.* call lands on the
        # GPU. The helper takes care of init() and the global backend.
        Nx.Vulkan.jit(fun, opts)

      compiler ->
        opts = translate_opts(compiler, opts)
        opts = force_host_if_no_gpu(compiler, opts)
        Nx.Defn.jit(fun, [{:compiler, compiler} | opts])
    end
  end

  @doc """
  Detect the best available JIT compiler module.

  Returns `EXLA`, `EMLX`, or `nil`. Respects `config :exmc, :compiler` override.
  """
  def detect_compiler do
    case Application.get_env(:exmc, :compiler) do
      nil -> auto_detect()
      :exla -> if loaded?(EXLA), do: EXLA, else: auto_detect()
      :emlx -> if loaded?(EMLX), do: EMLX, else: auto_detect()
      :vulkan -> if loaded?(Nx.Vulkan), do: Nx.Vulkan, else: auto_detect()
      :none -> nil
    end
  end

  @doc """
  Return the Nx backend module for the detected compiler.
  """
  def backend do
    case detect_compiler() do
      EXLA -> EXLA.Backend
      EMLX -> EMLX.Backend
      Nx.Vulkan -> Nx.Vulkan.Backend
      nil -> Nx.BinaryBackend
    end
  end

  @doc """
  Working float precision for the detected compiler.

  Returns `:f64` for EXLA/Evaluator, `:f32` for EMLX (Metal limitation).
  """
  def precision do
    case detect_compiler() do
      EMLX -> :f32
      # Vulkan compute shaders are f32-only; f64 storage round-trips for
      # mass-matrix accumulators but per-step ops drop to f32.
      Nx.Vulkan -> :f32
      _ -> :f64
    end
  end

  @doc """
  Downcast tensor to working precision if needed.

  When EMLX is active, f64 tensors are cast to f32. Otherwise returns unchanged.
  """
  def ensure_precision(%Nx.Tensor{} = t) do
    if precision() == :f32 and Nx.type(t) == {:f, 64} do
      Nx.as_type(t, :f32)
    else
      t
    end
  end

  def ensure_precision(other), do: other

  @doc """
  Check if the purpose-built MLX NIF is loaded and available.

  When true, `Exmc.MLX.Compiler` can bypass EMLX's Evaluator fallback
  for models without Custom distributions.
  """
  def mlx_nif_available? do
    Code.ensure_loaded?(Exmc.MLX.Native) and Exmc.MLX.Native.available?()
  end

  # --- Private ---

  defp auto_detect do
    cond do
      loaded?(EXLA) -> EXLA
      loaded?(EMLX) -> EMLX
      # Vulkan auto-picks when EXLA and EMLX are both absent — the FreeBSD
      # GPU path. EXLA still wins on hosts that have it (a CUDA-equipped
      # Linux box won't accidentally drop down to Vulkan).
      loaded?(Nx.Vulkan) -> Nx.Vulkan
      true -> nil
    end
  end

  defp loaded?(mod) do
    Code.ensure_loaded?(mod) and function_exported?(mod, :__info__, 1)
  end

  # When CUDA_VISIBLE_DEVICES="" (GPU hidden), force EXLA to use host client.
  # Without this, EXLA still attempts a CUDA client init which crashes
  # the EXLA.Client GenServer and cascades to all subsequent JIT calls.
  defp force_host_if_no_gpu(compiler, opts) when compiler == EXLA do
    if System.get_env("CUDA_VISIBLE_DEVICES") == "" and not Keyword.has_key?(opts, :client) do
      Keyword.put(opts, :client, :host)
    else
      opts
    end
  end

  defp force_host_if_no_gpu(_compiler, opts), do: opts

  # Translate EXLA-style opts to EMLX equivalents
  defp translate_opts(compiler, opts) when compiler == EMLX do
    case Keyword.pop(opts, :client) do
      {nil, opts} -> opts
      {:cuda, opts} -> [{:device, :gpu} | opts]
      {:host, opts} -> [{:device, :cpu} | opts]
      {_, opts} -> opts
    end
  end

  defp translate_opts(_compiler, opts), do: opts
end
