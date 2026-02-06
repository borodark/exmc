defmodule Exmc.Rewrite.Pass do
  @moduledoc """
  Behaviour for IR rewrite passes.
  """

  alias Exmc.IR

  @callback name() :: String.t()
  @callback run(ir :: IR.t()) :: IR.t()
end
