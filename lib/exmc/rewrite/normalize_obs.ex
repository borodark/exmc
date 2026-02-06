defmodule Exmc.Rewrite.NormalizeObs do
  @moduledoc """
  Normalize obs nodes to a canonical 4-tuple with metadata.
  """

  @behaviour Exmc.Rewrite.Pass

  alias Exmc.{IR, Node}

  @impl true
  def name, do: "normalize_obs"

  @impl true
  def run(%IR{} = ir) do
    nodes =
      ir.nodes
      |> Enum.map(fn {id, node} -> {id, rewrite_node(node)} end)
      |> Map.new()

    %{ir | nodes: nodes}
  end

  defp rewrite_node(%Node{op: {:obs, target_id, value}} = node) do
    %Node{node | op: {:obs, target_id, value, %{}}}
  end

  defp rewrite_node(node), do: node
end
