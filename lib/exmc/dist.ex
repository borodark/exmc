defmodule Exmc.Dist do
  @moduledoc """
  Distribution interface for logpdf and support/transform metadata.
  """

  @callback logpdf(value :: Nx.t(), params :: map()) :: Nx.t()
  @callback support(params :: map()) :: atom()
  @callback transform(params :: map()) :: atom() | nil
end
