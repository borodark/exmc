defmodule Exmc.Rewrite do
  @moduledoc """
  Rewrite pipeline for probabilistic IR.

  ## Examples

      iex> Exmc.Rewrite.pass_names() |> Enum.member?("attach_default_transforms")
      true
  """

  alias Exmc.IR

  @passes [
    Exmc.Rewrite.AttachDefaultTransforms,
    Exmc.Rewrite.LiftMeasurableMatmul,
    Exmc.Rewrite.LiftMeasurableAffine,
    Exmc.Rewrite.NormalizeObs,
    Exmc.Rewrite.PopulateObsMetadata
  ]

  def apply(%IR{} = ir) do
    Enum.reduce(@passes, ir, fn pass, acc -> pass.run(acc) end)
  end

  def passes, do: @passes

  def pass_names do
    Enum.map(@passes, & &1.name())
  end
end
