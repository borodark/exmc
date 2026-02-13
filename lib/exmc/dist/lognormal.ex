defmodule Exmc.Dist.Lognormal do
  @moduledoc """
  Lognormal distribution parameterized by mu and sigma of the underlying Normal.

  ## Examples

      iex> x = Nx.tensor(1.0)
      iex> Exmc.Dist.Lognormal.logpdf(x, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)}) |> Nx.to_number() |> Float.round(6)
      -0.918939
  """

  @behaviour Exmc.Dist

  @impl true
  def logpdf(x, %{mu: mu, sigma: sigma}) do
    safe_sigma = Nx.max(sigma, Nx.tensor(1.0e-30))
    log_x = Nx.log(x)
    z = Nx.divide(Nx.subtract(log_x, mu), safe_sigma)
    z2 = Nx.multiply(z, z)
    two_pi = Nx.tensor(2.0 * :math.pi())
    log_term = Nx.add(Nx.log(two_pi), Nx.multiply(Nx.tensor(2.0), Nx.log(safe_sigma)))

    Nx.multiply(Nx.tensor(-0.5), Nx.add(z2, log_term))
    |> Nx.subtract(log_x)
  end

  @impl true
  def support(_params), do: :positive

  @impl true
  def transform(_params), do: :log

  @impl true
  def sample(%{mu: mu, sigma: sigma}, rng) do
    mu_f = Nx.to_number(mu)
    sigma_f = Nx.to_number(sigma)
    {z, rng} = :rand.normal_s(rng)
    value = :math.exp(mu_f + sigma_f * z)
    {Nx.tensor(value), rng}
  end
end
