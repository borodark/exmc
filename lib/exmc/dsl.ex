defmodule Exmc.DSL do
  @moduledoc """
  Minimal DSL helpers for building IRs.

  ## Examples

      iex> alias Exmc.DSL
      iex> use Exmc.DSL
      iex> ir = DSL.model do
      ...>   rv("x", Exmc.Dist.Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
      ...>   obs("x_obs", "x", Nx.tensor(0.2))
      ...> end
      iex> map_size(ir.nodes) == 2
      true
  """

  defmacro model(do: block) do
    quote do
      var!(ir) = Exmc.Builder.new_ir()
      unquote(block)
      var!(ir)
    end
  end

  defmacro rv(id, dist, params, opts \\ []) do
    quote do
      var!(ir) = Exmc.Builder.rv(var!(ir), unquote(id), unquote(dist), unquote(params), unquote(opts))
    end
  end

  defmacro obs(id, target_id, value) do
    quote do
      var!(ir) = Exmc.Builder.obs(var!(ir), unquote(id), unquote(target_id), unquote(value))
    end
  end

  defmacro obs(id, target_id, value, opts) do
    quote do
      var!(ir) =
        Exmc.Builder.obs(var!(ir), unquote(id), unquote(target_id), unquote(value), unquote(opts))
    end
  end

  defmacro det(id, fun, args) do
    quote do
      var!(ir) = Exmc.Builder.det(var!(ir), unquote(id), unquote(fun), unquote(args))
    end
  end

  defmacro matmul(id, a, rv_id) do
    quote do
      var!(ir) = Exmc.Builder.det(var!(ir), unquote(id), :matmul, [unquote(a), unquote(rv_id)])
    end
  end

  defmacro affine(id, a, b, rv_id) do
    quote do
      var!(ir) =
        Exmc.Builder.det(var!(ir), unquote(id), :affine, [unquote(a), unquote(b), unquote(rv_id)])
    end
  end

  defmacro __using__(_opts) do
    quote do
      import Exmc.DSL
    end
  end
end
