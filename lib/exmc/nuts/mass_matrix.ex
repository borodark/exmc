defmodule Exmc.NUTS.MassMatrix do
  @moduledoc """
  Diagonal mass matrix adaptation via Welford online variance.

  The finalized variance IS the `inv_mass_diag` (standard choice: `M^{-1} = Var(q)`).
  """

  @doc """
  Initialize Welford state for dimension `d`.
  """
  def init(d) when is_integer(d) and d > 0 do
    %{
      n: 0,
      mean: Nx.broadcast(Nx.tensor(0.0, type: :f64), {d}),
      m2: Nx.broadcast(Nx.tensor(0.0, type: :f64), {d})
    }
  end

  @doc """
  Welford online update with a new sample `q` (flat f64 tensor).
  """
  def update(%{n: n, mean: mean, m2: m2}, q) do
    new_n = n + 1
    delta = Nx.subtract(q, mean)
    new_mean = Nx.add(mean, Nx.divide(delta, Nx.tensor(new_n * 1.0, type: :f64)))
    delta2 = Nx.subtract(q, new_mean)
    new_m2 = Nx.add(m2, Nx.multiply(delta, delta2))

    %{n: new_n, mean: new_mean, m2: new_m2}
  end

  @doc """
  Finalize: return `inv_mass_diag` as `max(variance, 1e-3)`.

  Returns identity (ones) if fewer than 3 samples.
  """
  def finalize(%{n: n, mean: mean}) when n < 3 do
    {d} = Nx.shape(mean)
    Nx.broadcast(Nx.tensor(1.0, type: :f64), {d})
  end

  def finalize(%{n: n, m2: m2}) do
    variance = Nx.divide(m2, Nx.tensor((n - 1) * 1.0, type: :f64))
    floor = Nx.tensor(1.0e-3, type: :f64)
    Nx.max(variance, floor)
  end

  @doc """
  Reset Welford state (same as init).
  """
  def reset(d) when is_integer(d) and d > 0, do: init(d)
end
