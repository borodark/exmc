defmodule Exmc.Compiler do
  @moduledoc """
  Compiles IR into a differentiable logp function.

  The compiler pre-dispatches at build time: it walks all IR nodes once,
  producing a list of term closures that are pure Nx ops at runtime. This
  means the returned `logp_fn` and `vag_fn` are ready for `Nx.Defn.grad`.
  """

  alias Exmc.{IR, Rewrite, PointMap, Transform}

  @doc """
  Compile an IR into a logp function and its PointMap.

  Returns `{logp_fn, point_map}` where `logp_fn :: flat_tensor -> scalar_logp`.
  """
  def compile(%IR{} = ir) do
    ir = Rewrite.apply(ir)
    pm = PointMap.build(ir)
    terms = build_terms(ir, pm)

    logp_fn =
      if pm.size == 0 do
        # No free RVs — logp is a constant (all obs terms eagerly computed)
        constant = eval_terms(terms, %{})
        fn _flat -> constant end
      else
        fn flat ->
          vm = PointMap.unpack(flat, pm)
          eval_terms(terms, vm)
        end
      end

    {logp_fn, pm}
  end

  @doc """
  Compile an IR into a value-and-grad function and its PointMap.

  Returns `{vag_fn, point_map}` where `vag_fn :: flat_tensor -> {logp, grad}`.
  """
  def value_and_grad(%IR{} = ir) do
    {logp_fn, pm} = compile(ir)

    vag_fn = fn flat ->
      Nx.Defn.value_and_grad(flat, logp_fn)
    end

    {vag_fn, pm}
  end

  # --- Private: term generation ---

  defp build_terms(%IR{} = ir, %PointMap{} = pm) do
    ir.nodes
    |> Map.values()
    |> Enum.flat_map(&node_term(&1, ir, pm))
  end

  # Free RV without transform
  defp node_term(%{id: id, op: {:rv, dist, params}}, _ir, pm) do
    if PointMap.has_entry?(pm, id) do
      [fn vm -> dist.logpdf(Map.fetch!(vm, id), params) end]
    else
      []
    end
  end

  # Free RV with transform
  defp node_term(%{id: id, op: {:rv, dist, params, transform}}, _ir, pm) do
    if PointMap.has_entry?(pm, id) do
      [
        fn vm ->
          z = Map.fetch!(vm, id)
          x = Transform.apply(transform, z)
          logp = dist.logpdf(x, params)
          jac = Transform.log_abs_det_jacobian(transform, z)
          Nx.add(logp, jac)
        end
      ]
    else
      []
    end
  end

  # Obs without meta -> delegate to obs with empty meta
  defp node_term(%{op: {:obs, target_id, value}} = node, ir, pm) do
    node_term(%{node | op: {:obs, target_id, value, %{}}}, ir, pm)
  end

  # Obs with meta — eagerly compute constant logp
  defp node_term(%{op: {:obs, target_id, value, meta}}, ir, _pm) do
    if Map.get(meta, :likelihood, true) == false do
      []
    else
      target_node = IR.get_node!(ir, target_id)
      eager_obs_term(target_node, value, meta)
    end
  end

  # Meas obs without meta
  defp node_term(%{op: {:meas_obs, rv_id, value, op_info}} = node, ir, pm) do
    node_term(%{node | op: {:meas_obs, rv_id, value, op_info, %{}}}, ir, pm)
  end

  # Meas obs with meta — eagerly compute
  defp node_term(%{op: {:meas_obs, rv_id, value, op_info, meta}}, ir, _pm) do
    rv_node = IR.get_node!(ir, rv_id)
    eager_meas_obs_term(rv_node, value, op_info, meta)
  end

  # Det and other nodes contribute nothing
  defp node_term(_node, _ir, _pm), do: []

  # --- Eager obs computation (constant w.r.t. free RVs) ---

  defp eager_obs_term(%{op: {:rv, dist, params}}, value, meta) do
    logp = dist.logpdf(value, params)
    [fn _vm -> apply_obs_meta(logp, meta) end]
  end

  defp eager_obs_term(%{op: {:rv, dist, params, transform}}, value, meta) do
    z = inverse_transform(transform, value)
    x = Transform.apply(transform, z)
    logp = dist.logpdf(x, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_obs_term(_target, _value, _meta), do: []

  # --- Eager meas_obs computation ---

  defp eager_meas_obs_term(%{op: {:rv, dist, params}}, value, {:matmul, a}, meta) do
    x = Nx.LinAlg.solve(a, value)
    logp = dist.logpdf(x, params)
    jac = Nx.negate(Nx.log(Nx.abs(Nx.LinAlg.determinant(a))))
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params, transform}}, value, {:matmul, a}, meta) do
    x = Nx.LinAlg.solve(a, value)
    z = inverse_transform(transform, x)
    x2 = Transform.apply(transform, z)
    logp = dist.logpdf(x2, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    meas_jac = Nx.negate(Nx.log(Nx.abs(Nx.LinAlg.determinant(a))))
    combined = Nx.add(Nx.add(logp, jac), meas_jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params}}, value, {:affine, a, b}, meta) do
    a_t = to_tensor(a)
    b_t = to_tensor(b)
    x = Nx.divide(Nx.subtract(value, b_t), a_t)
    logp = dist.logpdf(x, params)
    jac = Nx.negate(Nx.log(Nx.abs(a_t)))
    combined = Nx.add(logp, jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(%{op: {:rv, dist, params, transform}}, value, {:affine, a, b}, meta) do
    a_t = to_tensor(a)
    b_t = to_tensor(b)
    x = Nx.divide(Nx.subtract(value, b_t), a_t)
    z = inverse_transform(transform, x)
    x2 = Transform.apply(transform, z)
    logp = dist.logpdf(x2, params)
    jac = Transform.log_abs_det_jacobian(transform, z)
    meas_jac = Nx.negate(Nx.log(Nx.abs(a_t)))
    combined = Nx.add(Nx.add(logp, jac), meas_jac)
    [fn _vm -> apply_obs_meta(combined, meta) end]
  end

  defp eager_meas_obs_term(_rv_node, _value, _op_info, _meta), do: []

  # --- Term evaluation ---

  defp eval_terms([], _vm), do: Nx.tensor(0.0)

  defp eval_terms(terms, vm) do
    terms
    |> Enum.map(fn term -> term.(vm) end)
    |> sum_logps()
  end

  defp sum_logps([one]), do: Nx.reshape(one, {})
  defp sum_logps(list), do: Enum.reduce(list, fn x, acc -> Nx.add(Nx.reshape(x, {}), acc) end)

  # --- Helpers ---

  defp apply_obs_meta(logp, meta) do
    weight = Map.get(meta, :weight, 1.0)
    weight_t = to_tensor(weight)
    logp = Nx.multiply(logp, weight_t)

    masked =
      case Map.get(meta, :mask) do
        nil -> logp
        mask -> Nx.select(to_tensor(mask), logp, Nx.tensor(0.0))
      end

    case Map.get(meta, :reduce) do
      :sum -> Nx.sum(masked)
      :mean -> Nx.mean(masked)
      :logsumexp -> Nx.logsumexp(masked)
      _ -> masked
    end
  end

  defp inverse_transform(nil, x), do: x
  defp inverse_transform(:log, x), do: Nx.log(x)
  defp inverse_transform(:softplus, x), do: Nx.log(Nx.expm1(x))
  defp inverse_transform(:logit, x), do: Nx.subtract(Nx.log(x), Nx.log1p(Nx.negate(x)))

  defp to_tensor(%Nx.Tensor{} = t), do: t
  defp to_tensor(v) when is_number(v) or is_boolean(v), do: Nx.tensor(v)
end
