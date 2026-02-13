defmodule Exmc.MvNormalTest do
  use ExUnit.Case, async: true

  alias Exmc.{Builder, Compiler, PointMap, Rewrite}
  alias Exmc.Dist.MvNormal
  import Exmc.TestHelper

  # --- logpdf tests ---

  test "2d standard MvNormal at origin" do
    mu = Nx.tensor([0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = Nx.tensor([0.0, 0.0])

    result = MvNormal.logpdf(x, %{mu: mu, cov: cov}) |> Nx.to_number()
    # -0.5 * (2 * log(2pi) + 0 + 0) = -log(2pi)
    expected = -:math.log(2.0 * :math.pi())
    assert_in_delta result, expected, 1.0e-6
  end

  test "2d standard MvNormal at (1, 0)" do
    mu = Nx.tensor([0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
    x = Nx.tensor([1.0, 0.0])

    result = MvNormal.logpdf(x, %{mu: mu, cov: cov}) |> Nx.to_number()
    # -0.5 * (2*log(2pi) + 0 + 1) = -log(2pi) - 0.5
    expected = -:math.log(2.0 * :math.pi()) - 0.5
    assert_in_delta result, expected, 1.0e-6
  end

  test "3d MvNormal with known covariance" do
    mu = Nx.tensor([1.0, 2.0, 3.0])
    cov = Nx.tensor([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
    x = Nx.tensor([1.0, 2.0, 3.0])

    result = MvNormal.logpdf(x, %{mu: mu, cov: cov}) |> Nx.to_number()
    # At mean: mahal = 0, so logpdf = -0.5 * (3*log(2pi) + log|cov|)
    # |cov| = det([[2, 0.5, 0], [0.5, 1, 0], [0, 0, 3]]) = 3 * (2*1 - 0.25) = 3 * 1.75 = 5.25
    expected = -0.5 * (3.0 * :math.log(2.0 * :math.pi()) + :math.log(5.25))
    assert_in_delta result, expected, 1.0e-5
  end

  test "MvNormal support and transform" do
    params = %{mu: Nx.tensor([0.0, 0.0]), cov: Nx.tensor([[1.0, 0.0], [0.0, 1.0]])}
    assert MvNormal.support(params) == :real
    assert MvNormal.transform(params) == nil
  end

  test "MvNormal sample produces correct shape" do
    mu = Nx.tensor([0.0, 0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rng = :rand.seed_s(:exsss, 42)

    {sample, _rng} = MvNormal.sample(%{mu: mu, cov: cov}, rng)
    assert Nx.shape(sample) == {3}
  end

  # --- Builder + PointMap integration ---

  test "MvNormal in Builder with shape" do
    mu = Nx.tensor([0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", MvNormal, %{mu: mu, cov: cov}, shape: {2})
      |> Rewrite.apply()

    pm = PointMap.build(ir)

    assert pm.size == 2
    assert length(pm.entries) == 1
    entry = hd(pm.entries)
    assert entry.id == "x"
    assert entry.length == 2
    assert entry.shape == {2}
    assert entry.transform == nil
  end

  test "MvNormal compile + logp" do
    mu = Nx.tensor([0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", MvNormal, %{mu: mu, cov: cov}, shape: {2})

    {logp_fn, pm} = Compiler.compile(ir)

    x_val = Nx.tensor([0.0, 0.0])
    flat = PointMap.pack(%{"x" => x_val}, pm)
    compiled_logp = logp_fn.(flat) |> Nx.to_number()

    expected = -:math.log(2.0 * :math.pi())
    assert_in_delta compiled_logp, expected, 1.0e-5
  end

  test "MvNormal gradient via value_and_grad" do
    mu = Nx.tensor([0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", MvNormal, %{mu: mu, cov: cov}, shape: {2})

    {vag_fn, pm} = Compiler.value_and_grad(ir)

    x_val = Nx.tensor([0.5, -0.3])
    flat = PointMap.pack(%{"x" => x_val}, pm)
    {logp, grad} = vag_fn.(flat)

    # For standard MvNormal: grad_logp = -(x - mu) = -x
    expected_grad = Nx.negate(x_val)
    assert_close(grad, expected_grad, 1.0e-4)

    # Verify logp value
    direct_logp = MvNormal.logpdf(x_val, %{mu: mu, cov: cov}) |> Nx.to_number()
    assert_in_delta Nx.to_number(logp), direct_logp, 1.0e-5
  end

  test "MvNormal pack/unpack roundtrip" do
    mu = Nx.tensor([0.0, 0.0, 0.0])
    cov = Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    ir =
      Builder.new_ir()
      |> Builder.rv("x", MvNormal, %{mu: mu, cov: cov}, shape: {3})
      |> Rewrite.apply()

    pm = PointMap.build(ir)

    x_val = Nx.tensor([1.0, 2.0, 3.0])
    flat = PointMap.pack(%{"x" => x_val}, pm)
    recovered = PointMap.unpack(flat, pm)

    assert_close(recovered["x"], x_val)
  end
end
