defmodule Exmc.Dist.Weibull do
  @moduledoc """
  Weibull distribution with shape k (> 0) and scale lambda (> 0).

  PDF: f(t; k, lambda) = (k/lambda) * (t/lambda)^(k-1) * exp(-(t/lambda)^k)

  ## Examples

      iex> t = Nx.tensor(1.0)
      iex> Exmc.Dist.Weibull.logpdf(t, %{k: Nx.tensor(2.0), lambda: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -0.306853
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(t, %{k: k, lambda: lambda}) do
    log_lambda = Nx.log(lambda)
    log_k = Nx.log(k)
    # z = log(t/lambda)
    z = Nx.subtract(Nx.log(t), log_lambda)
    # log(k) - log(lambda) + (k-1)*log(t/lambda) - (t/lambda)^k
    Nx.subtract(
      Nx.add(Nx.subtract(log_k, log_lambda), Nx.multiply(Nx.subtract(k, 1.0), z)),
      Nx.exp(Nx.multiply(k, z))
    )
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :log

  @impl true
  def sample(%{k: k, lambda: lambda}, rng) do
    k_f = Nx.to_number(k)
    lambda_f = Nx.to_number(lambda)
    {u, rng} = :rand.uniform_s(rng)
    # Inverse CDF: t = lambda * (-log(U))^(1/k)
    value = lambda_f * :math.pow(-:math.log(u), 1.0 / k_f)
    {Nx.tensor(value), rng}
  end

  @doc """
  Log survival function: log(1 - CDF(t)) = -(t/lambda)^k

  Used by Censored module for right-censored observations.
  """
  def log_survival(t, %{k: k, lambda: lambda}) do
    z = Nx.subtract(Nx.log(t), Nx.log(lambda))
    Nx.negate(Nx.exp(Nx.multiply(k, z)))
  end
end
