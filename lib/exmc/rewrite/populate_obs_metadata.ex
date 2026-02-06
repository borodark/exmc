defmodule Exmc.Rewrite.PopulateObsMetadata do
  @moduledoc """
  Populate obs metadata with defaults or derived values.

  Fields:
  - :likelihood (default true)
  - :weight (default 1.0)
  - :mask (default nil)
  - :reduce (default nil)
  """

  @behaviour Exmc.Rewrite.Pass

  alias Exmc.{IR, Node}

  @impl true
  def name, do: "populate_obs_metadata"

  @impl true
  def run(%IR{} = ir) do
    nodes =
      ir.nodes
      |> Enum.map(fn {id, node} -> {id, rewrite_node(node)} end)
      |> Map.new()

    %{ir | nodes: nodes}
  end

  defp rewrite_node(%Node{op: {:obs, target_id, value, meta}} = node) when is_map(meta) do
    meta =
      meta
      |> Map.put_new(:likelihood, true)
      |> Map.put_new(:weight, 1.0)
      |> Map.put_new(:mask, nil)
      |> Map.put_new(:reduce, nil)

    %Node{node | op: {:obs, target_id, value, meta}}
  end

  defp rewrite_node(node), do: node
end
