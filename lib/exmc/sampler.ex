defmodule Exmc.Sampler do
  @moduledoc """
  Convenience facade for `Exmc.NUTS.Sampler`.

  Delegates all public functions so notebooks and user code can write
  `Exmc.Sampler.sample(...)` instead of `Exmc.NUTS.Sampler.sample(...)`.
  """

  defdelegate sample(ir, init_values \\ %{}, opts \\ []), to: Exmc.NUTS.Sampler
  defdelegate compile(ir, opts \\ []), to: Exmc.NUTS.Sampler
  defdelegate sample_compiled(compiled, init_values \\ %{}, opts \\ []), to: Exmc.NUTS.Sampler
  defdelegate sample_chains(ir, init_values, opts), to: Exmc.NUTS.Sampler
end
