defmodule Exmc.Benchmark.ReliabilityModel do
  @moduledoc """
  Hierarchical Weibull reliability model builder for eXMC.

  Builds a d=44 model: 4 hyperparameters + 20 types × 2 NCP params.
  Uses Weibull log-likelihood with right-censoring support.
  """

  def build(data) do
    alias Exmc.{Builder, Dist}

    n_types = data.n_types

    # Flatten all observations into tensors
    all_times =
      data.obs_by_type
      |> Enum.flat_map(fn type_data -> type_data.times end)
      |> Nx.tensor(type: :f64)

    all_censored =
      data.obs_by_type
      |> Enum.flat_map(fn type_data -> Enum.map(type_data.censored, fn c -> if c, do: 1.0, else: 0.0 end) end)
      |> Nx.tensor(type: :f64)

    # Type indices (which type each observation belongs to)
    type_indices =
      data.obs_by_type
      |> Enum.with_index()
      |> Enum.flat_map(fn {type_data, idx} -> List.duplicate(idx * 1.0, type_data.n_obs) end)
      |> Nx.tensor(type: :f64)

    obs_data = Nx.stack([all_times, all_censored, type_indices], axis: 1)

    ir =
      Builder.new_ir()
      |> Builder.data(obs_data)
      |> Builder.rv("log_k_mean", Dist.Normal, %{mu: Nx.tensor(0.5, type: :f64), sigma: Nx.tensor(1.0, type: :f64)})
      |> Builder.rv("log_k_sigma", Dist.HalfCauchy, %{scale: Nx.tensor(1.0, type: :f64)})
      |> Builder.rv("log_l_mean", Dist.Normal, %{mu: Nx.tensor(2.0, type: :f64), sigma: Nx.tensor(1.0, type: :f64)})
      |> Builder.rv("log_l_sigma", Dist.HalfCauchy, %{scale: Nx.tensor(1.0, type: :f64)})

    # NCP params for each type
    ir =
      Enum.reduce(0..(n_types - 1), ir, fn j, ir ->
        ir
        |> Builder.rv("k_raw_#{j}", Dist.Normal, %{mu: Nx.tensor(0.0, type: :f64), sigma: Nx.tensor(1.0, type: :f64)})
        |> Builder.rv("l_raw_#{j}", Dist.Normal, %{mu: Nx.tensor(0.0, type: :f64), sigma: Nx.tensor(1.0, type: :f64)})
      end)

    # Custom likelihood: hierarchical Weibull with censoring
    n_types_val = n_types
    n_per_type_val = data.n_per_type

    logpdf_fn = fn _x, params ->
      data_tensor = params.__obs_data
      times = data_tensor[[.., 0]]
      censored = data_tensor[[.., 1]]
      type_idx = data_tensor[[.., 2]]

      log_k_mean = params.log_k_mean
      log_k_sigma = Nx.max(params.log_k_sigma, Nx.tensor(0.01, type: :f64))
      log_l_mean = params.log_l_mean
      log_l_sigma = Nx.max(params.log_l_sigma, Nx.tensor(0.01, type: :f64))

      # Build per-type k and lambda from NCP
      n = n_types_val
      n_per = n_per_type_val

      # For each observation, compute its type's k and lambda
      # Simple approach: compute all types, then index
      logp =
        Enum.reduce(0..(n - 1), Nx.tensor(0.0, type: :f64), fn j, acc ->
          k_raw = Map.get(params, :"k_raw_#{j}")
          l_raw = Map.get(params, :"l_raw_#{j}")

          k = Nx.exp(Nx.add(log_k_mean, Nx.multiply(log_k_sigma, k_raw)))
          lambda = Nx.exp(Nx.add(log_l_mean, Nx.multiply(log_l_sigma, l_raw)))

          # Slice this type's observations
          start_idx = j * n_per
          t_j = Nx.slice(times, [start_idx], [n_per])
          c_j = Nx.slice(censored, [start_idx], [n_per])

          # Weibull log-likelihood (numerically stable):
          # Uncensored: log(k) - log(lambda) + (k-1)*log(t/lambda) - (t/lambda)^k
          # Censored (survival): -(t/lambda)^k
          #
          # Compute (t/lambda)^k in log-space to prevent overflow:
          # (t/lambda)^k = exp(k * log(t/lambda))
          # Clamp the exponent to [-500, 500] to prevent Inf/NaN
          t_safe = Nx.max(t_j, Nx.tensor(0.01, type: :f64))
          k_safe = Nx.max(k, Nx.tensor(0.1, type: :f64))
          lambda_safe = Nx.max(lambda, Nx.tensor(0.01, type: :f64))

          log_t_norm = Nx.log(Nx.divide(t_safe, lambda_safe))
          log_t_norm_k = Nx.multiply(k_safe, log_t_norm)
          # Clamp to prevent exp overflow
          log_t_norm_k_clamped = Nx.min(Nx.max(log_t_norm_k, Nx.tensor(-500.0, type: :f64)), Nx.tensor(500.0, type: :f64))
          t_norm_k = Nx.exp(log_t_norm_k_clamped)

          # Full Weibull log-pdf: log(k/lambda) + (k-1)*log(t/lambda) - (t/lambda)^k
          log_pdf =
            Nx.log(k_safe)
            |> Nx.subtract(Nx.log(lambda_safe))
            |> Nx.add(Nx.multiply(Nx.subtract(k_safe, Nx.tensor(1.0, type: :f64)), log_t_norm))
            |> Nx.subtract(t_norm_k)
          log_surv = Nx.negate(t_norm_k)

          # Mix: (1 - censored) * log_pdf + censored * log_surv
          contrib = Nx.add(
            Nx.multiply(Nx.subtract(Nx.tensor(1.0, type: :f64), c_j), log_pdf),
            Nx.multiply(c_j, log_surv)
          )

          Nx.add(acc, Nx.sum(contrib))
        end)

      logp
    end

    dist = Dist.Custom.new(logpdf_fn, support: :real)

    # Build param map for custom likelihood
    param_refs = %{
      log_k_mean: "log_k_mean", log_k_sigma: "log_k_sigma",
      log_l_mean: "log_l_mean", log_l_sigma: "log_l_sigma",
      __obs_data: "__obs_data"
    }

    param_refs =
      Enum.reduce(0..(n_types - 1), param_refs, fn j, refs ->
        refs
        |> Map.put(:"k_raw_#{j}", "k_raw_#{j}")
        |> Map.put(:"l_raw_#{j}", "l_raw_#{j}")
      end)

    ir =
      Dist.Custom.rv(ir, "lik", dist, param_refs)
      |> Builder.obs("lik_obs", "lik", Nx.tensor(0.0, type: :f64))

    ir
  end

  def init_values(data) do
    init = %{
      "log_k_mean" => 0.5,
      "log_k_sigma" => 0.3,
      "log_l_mean" => 2.0,
      "log_l_sigma" => 0.3
    }

    Enum.reduce(0..(data.n_types - 1), init, fn j, acc ->
      acc
      |> Map.put("k_raw_#{j}", 0.0)
      |> Map.put("l_raw_#{j}", 0.0)
    end)
  end

  def reconstruct_params(trace, data) do
    log_k_mean = Nx.to_flat_list(trace["log_k_mean"])
    log_k_sigma = Nx.to_flat_list(trace["log_k_sigma"])
    log_l_mean = Nx.to_flat_list(trace["log_l_mean"])
    log_l_sigma = Nx.to_flat_list(trace["log_l_sigma"])

    hyper = %{
      "log_k_mean" => trace["log_k_mean"],
      "log_k_sigma" => trace["log_k_sigma"],
      "log_l_mean" => trace["log_l_mean"],
      "log_l_sigma" => trace["log_l_sigma"]
    }

    hyper
  end
end
