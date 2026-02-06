defmodule Exmc.Rewrite.LiftMeasurableMatmul do
  @moduledoc """
  Rewrite obs(det(matmul(A, rv))) into a measurable observation.
  """

  @behaviour Exmc.Rewrite.Pass

  alias Exmc.{IR, Node}

  @impl true
  def name, do: "lift_measurable_matmul"

  @impl true
  def run(%IR{} = ir) do
    nodes =
      ir.nodes
      |> Enum.map(fn {id, node} -> {id, rewrite_node(node, ir)} end)
      |> Map.new()

    %{ir | nodes: nodes}
  end

  defp rewrite_node(%Node{op: {:obs, target_id, value}} = node, %IR{} = ir) do
    target = IR.get_node!(ir, target_id)

    case target.op do
      {:det, :matmul, [a, rv_id]} when is_binary(rv_id) ->
        %Node{node | op: {:meas_obs, rv_id, value, {:matmul, a}, %{}}}

      _ ->
        node
    end
  end

  defp rewrite_node(%Node{op: {:obs, target_id, value, meta}} = node, %IR{} = ir)
       when is_map(meta) do
    target = IR.get_node!(ir, target_id)

    case target.op do
      {:det, :matmul, [a, rv_id]} when is_binary(rv_id) ->
        %Node{node | op: {:meas_obs, rv_id, value, {:matmul, a}, meta}}

      _ ->
        node
    end
  end

  defp rewrite_node(node, _ir), do: node
end
