defmodule Exmc.Rewrite.AttachDefaultTransforms do
  @moduledoc """
  Attach default transforms to RV nodes based on distribution metadata.
  """

  @behaviour Exmc.Rewrite.Pass

  alias Exmc.{IR, Node}

  @impl true
  def name, do: "attach_default_transforms"

  @impl true
  def run(%IR{} = ir) do
    nodes =
      ir.nodes
      |> Enum.map(fn {id, node} -> {id, rewrite_node(node)} end)
      |> Map.new()

    %{ir | nodes: nodes}
  end

  defp rewrite_node(%Node{op: {:rv, dist, params}} = node) do
    case dist.transform(params) do
      nil -> node
      transform -> %Node{node | op: {:rv, dist, params, transform}}
    end
  end

  defp rewrite_node(node), do: node
end
