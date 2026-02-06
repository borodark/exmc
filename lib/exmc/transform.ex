defmodule Exmc.Transform do
  @moduledoc """
  Minimal transform support with log-abs-det Jacobian.

  ## Examples

      iex> z = Nx.tensor(0.0)
      iex> Exmc.Transform.apply(:log, z) |> Nx.to_number() |> Float.round(6)
      1.0
      iex> Exmc.Transform.log_abs_det_jacobian(:log, z) |> Nx.to_number() |> Float.round(6)
      0.0
  """

  def apply(nil, z), do: z

  def apply(:log, z) do
    Nx.exp(z)
  end

  def apply(:softplus, z) do
    Nx.log1p(Nx.exp(z))
  end

  def apply(:logit, z) do
    Nx.sigmoid(z)
  end

  def log_abs_det_jacobian(nil, _z), do: Nx.tensor(0.0)

  def log_abs_det_jacobian(:log, z) do
    # x = exp(z), |dx/dz| = exp(z), log|dx/dz| = z
    z
  end

  def log_abs_det_jacobian(:softplus, z) do
    # x = softplus(z), dx/dz = sigmoid(z)
    Nx.log(Nx.sigmoid(z))
  end

  def log_abs_det_jacobian(:logit, z) do
    # x = sigmoid(z), dx/dz = sigmoid(z) * (1 - sigmoid(z))
    s = Nx.sigmoid(z)
    Nx.add(Nx.log(s), Nx.log1p(Nx.negate(s)))
  end
end
