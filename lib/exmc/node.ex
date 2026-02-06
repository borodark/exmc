defmodule Exmc.Node do
  @moduledoc """
  IR node: random variables, observations, and deterministic ops.
  """

  defstruct [:id, :op, :deps, :shape, :dtype]

  @type t :: %__MODULE__{
          id: String.t(),
          op: term(),
          deps: [String.t()],
          shape: term(),
          dtype: term()
        }
end
