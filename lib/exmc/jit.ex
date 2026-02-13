defmodule Exmc.JIT do
  @moduledoc """
  Runtime JIT backend abstraction.

  Auto-detects available compilers in priority order: EXLA > EMLX > Evaluator.

  - **EXLA**: CUDA/ROCm/CPU acceleration, f64 supported. Default on Linux.
  - **EMLX**: Metal GPU acceleration on macOS via MLX, f32 only. Default on macOS
    when EXLA is not installed.
  - **Evaluator**: Pure Elixir fallback (BinaryBackend). Very slow but always works.

  ## Configuration

  Override auto-detection via application config:

      config :exmc, :compiler, :emlx   # force EMLX
      config :exmc, :compiler, :exla   # force EXLA
      config :exmc, :compiler, :none   # disable JIT

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

      compiler ->
        opts = translate_opts(compiler, opts)
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
      nil -> Nx.BinaryBackend
    end
  end

  @doc """
  Working float precision for the detected compiler.

  Returns `:f64` for EXLA/Evaluator, `:f32` for EMLX (Metal limitation).
  """
  def precision do
    if detect_compiler() == EMLX, do: :f32, else: :f64
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

  # --- Private ---

  defp auto_detect do
    cond do
      loaded?(EXLA) -> EXLA
      loaded?(EMLX) -> EMLX
      true -> nil
    end
  end

  defp loaded?(mod) do
    Code.ensure_loaded?(mod) and function_exported?(mod, :__info__, 1)
  end

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
