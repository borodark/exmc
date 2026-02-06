defmodule Exmc.NUTS.Leapfrog do
  @moduledoc """
  Leapfrog integrator and momentum utilities for HMC/NUTS.

  All position/momentum/gradient tensors are flat f64 vectors in unconstrained space.
  """

  @doc """
  One leapfrog step: half-momentum, full-position, evaluate, half-momentum.

  Returns `{q_new, p_new, logp_new, grad_new}`.
  """
  def step(vag_fn, q, p, grad, epsilon, inv_mass_diag) do
    eps = Nx.tensor(epsilon, type: :f64)
    half_eps = Nx.divide(eps, Nx.tensor(2.0, type: :f64))

    # half step for momentum
    p_half = Nx.add(p, Nx.multiply(half_eps, grad))

    # full step for position
    q_new = Nx.add(q, Nx.multiply(eps, Nx.multiply(inv_mass_diag, p_half)))

    # evaluate logp and gradient at new position
    {logp_new, grad_new} = vag_fn.(q_new)

    # half step for momentum
    p_new = Nx.add(p_half, Nx.multiply(half_eps, grad_new))

    {q_new, p_new, logp_new, grad_new}
  end

  @doc """
  Kinetic energy: `0.5 * sum(p^2 * inv_mass_diag)`.

  Returns a scalar Nx tensor.
  """
  def kinetic_energy(p, inv_mass_diag) do
    Nx.multiply(Nx.tensor(0.5, type: :f64), Nx.sum(Nx.multiply(Nx.multiply(p, p), inv_mass_diag)))
  end

  @doc """
  Joint log probability (negative Hamiltonian): `logp - kinetic_energy(p, inv_mass_diag)`.

  Higher is better. Returns a scalar Nx tensor.
  """
  def joint_logp(logp, p, inv_mass_diag) do
    Nx.subtract(logp, kinetic_energy(p, inv_mass_diag))
  end

  @doc """
  Sample momentum from the mass matrix.

  `p = z / sqrt(inv_mass_diag)` where `z ~ N(0, I)`.

  Returns `{p, new_key}`.
  """
  def sample_momentum(key, inv_mass_diag) do
    shape = Nx.shape(inv_mass_diag)
    {z, key} = Nx.Random.normal(key, shape: shape, type: :f64)
    p = Nx.divide(z, Nx.sqrt(inv_mass_diag))
    {p, key}
  end
end
