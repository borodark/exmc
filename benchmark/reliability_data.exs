defmodule Exmc.Benchmark.ReliabilityData do
  @moduledoc """
  Synthetic hierarchical Weibull reliability data generator.

  Generates failure times for `n_types` component types, each with
  its own Weibull shape (k) and scale (lambda). 20% of observations
  are right-censored (still running at observation time).
  """

  def generate(opts \\ []) do
    seed = Keyword.get(opts, :seed, 42)
    n_types = Keyword.get(opts, :n_types, 20)
    n_per_type = Keyword.get(opts, :n_per_type, 50)

    :rand.seed(:exsss, {seed, seed * 7 + 1, seed * 13 + 3})

    # Fleet-level hyperparameters
    log_k_mean = 0.4     # mean shape ~1.5 (wear-out)
    log_k_sigma = 0.3
    log_l_mean = 2.5     # mean scale ~12
    log_l_sigma = 0.3

    # Generate per-type parameters
    true_shapes = for _ <- 1..n_types do
      :math.exp(log_k_mean + :rand.normal() * log_k_sigma)
    end

    true_scales = for _ <- 1..n_types do
      :math.exp(log_l_mean + :rand.normal() * log_l_sigma)
    end

    # Generate observations per type
    censor_frac = 0.2
    obs_by_type =
      Enum.zip(true_shapes, true_scales)
      |> Enum.with_index()
      |> Enum.map(fn {{k, lambda}, _idx} ->
        times = for _ <- 1..n_per_type do
          # Weibull sample: t = lambda * (-ln(U))^(1/k)
          u = :rand.uniform()
          lambda * :math.pow(-:math.log(u), 1.0 / k)
        end

        # Right-censor some observations
        n_cens = trunc(n_per_type * censor_frac)
        {uncensored, to_censor} = Enum.split(times, n_per_type - n_cens)

        censored_times = Enum.map(to_censor, fn t ->
          # Censor at random fraction of failure time
          t * (0.3 + :rand.uniform() * 0.5)
        end)

        %{
          times: uncensored ++ censored_times,
          censored: List.duplicate(false, length(uncensored)) ++ List.duplicate(true, length(censored_times)),
          n_obs: n_per_type,
          n_cens: n_cens
        }
      end)

    %{
      n_types: n_types,
      n_per_type: n_per_type,
      true_shapes: true_shapes,
      true_scales: true_scales,
      obs_by_type: obs_by_type,
      hyper: %{log_k_mean: log_k_mean, log_k_sigma: log_k_sigma,
               log_l_mean: log_l_mean, log_l_sigma: log_l_sigma}
    }
  end
end
