# State-Space Models with eXMC

A practical guide to time series modeling using NUTS/HMC on the BEAM.

## What Is a State-Space Model?

Every state-space model has two equations:

```
State:       x_{t+1} = g(x_t, θ) + η_t       (how the hidden state evolves)
Observation: y_t     = f(x_t, θ) + ε_t        (how we see the state, noisily)
```

The **state** `x_t` is latent — you never observe it directly. You observe
`y_t` and infer what `x_t` must have been. The Bayesian approach: put priors
on everything (`x_{1:T}`, `θ`, noise variances), let NUTS find the joint
posterior.

## The eXMC Pattern

Every state-space model in eXMC follows the same four-step recipe:

```elixir
ir = Builder.new_ir()
  |> Builder.data(y)                                  # 1. Register observations
  |> Builder.rv("params", Dist, %{...})               # 2. Priors on parameters
  |> Builder.rv("state", GaussianRandomWalk, %{...})  # 3. Latent state process
  |> then(fn ir ->                                     # 4. Observation likelihood
    dist = Custom.new(fn _x, params -> ... end, support: :real)
    Custom.rv(ir, "lik", dist, %{...})
    |> Builder.obs("lik_obs", "lik", Nx.tensor(0.0, type: :f64))
  end)
```

Step 3 is the key insight: `GaussianRandomWalk` IS the state equation.
It says `x_t = x_{t-1} + η_t` where `η ~ N(0, σ²)`. NUTS samples the
entire state trajectory `x_{1:T}` jointly with the parameters.

## Model 1: Local Level (Trend + Noise)

The simplest state-space model. One latent state (trend), one observation
equation.

```
trend_{t+1} = trend_t + η_t,    η ~ N(0, σ_trend²)
y_t = trend_t + ε_t,            ε ~ N(0, σ_obs²)
```

Three free quantities: `σ_trend`, `σ_obs`, and the entire `trend_{1:T}` vector.

```elixir
alias Exmc.{Builder, Sampler}
alias Exmc.Dist.{Normal, HalfNormal, GaussianRandomWalk, Custom}

defmodule SSM do
  def t(v), do: Nx.tensor(v, type: :f64)

  def normal_logpdf_vec(x, mu, sigma) do
    z = Nx.divide(Nx.subtract(x, mu), sigma)
    Nx.sum(Nx.subtract(Nx.multiply(t(-0.5), Nx.multiply(z, z)), Nx.log(sigma)))
  end
end

# Data: 100 observations
y = Nx.tensor(your_data, type: :f64)
n = 100

# Build model
ir = Builder.new_ir()
  |> Builder.data(y)
  |> Builder.rv("sigma_trend", HalfNormal, %{sigma: SSM.t(1.0)})
  |> Builder.rv("sigma_obs", HalfNormal, %{sigma: SSM.t(1.0)})
  |> Builder.rv("trend", GaussianRandomWalk, %{sigma: "sigma_trend"}, shape: {n})

logpdf_fn = fn _x, params ->
  obs = params.__obs_data
  SSM.normal_logpdf_vec(obs, params.trend, params.sigma_obs)
end

dist = Custom.new(logpdf_fn, support: :real)

ir = Custom.rv(ir, "lik", dist, %{
    trend: "trend", sigma_obs: "sigma_obs", __obs_data: "__obs_data"
  })
  |> Builder.obs("lik_obs", "lik", Nx.tensor(0.0, type: :f64))

# Sample
{trace, stats} = Sampler.sample(ir,
  %{"sigma_trend" => 0.5, "sigma_obs" => 1.0, "trend" => Nx.tensor(your_data, type: :f64)},
  num_warmup: 1000, num_samples: 1000)

# Results
trend_mean = trace["trend"] |> Nx.mean(axes: [0])  # smoothed trend
sigma_trend = trace["sigma_trend"] |> Nx.mean() |> Nx.to_number()
sigma_obs = trace["sigma_obs"] |> Nx.mean() |> Nx.to_number()
```

**What you get**: The posterior mean of `trend` is a Bayesian smoother — like
an HP filter but with the smoothing parameter estimated from data instead of
chosen by convention.

## Model 2: Stochastic Volatility

The observation noise itself changes over time. Essential for financial data,
any series with crisis periods.

```
trend_{t+1} = trend_t + η_t,             η ~ N(0, σ_trend²)
h_{t+1} = h_t + γ_t,                     γ ~ N(0, σ_h²)
y_t = trend_t + exp(h_t / 2) · ε_t,      ε ~ N(0, 1)
```

Two latent state processes: trend and log-volatility.

```elixir
ir = Builder.new_ir()
  |> Builder.data(y)
  |> Builder.rv("sigma_trend", HalfNormal, %{sigma: SSM.t(2.0)})
  |> Builder.rv("sigma_h", HalfNormal, %{sigma: SSM.t(0.5)})
  |> Builder.rv("trend", GaussianRandomWalk, %{sigma: "sigma_trend"}, shape: {n})
  |> Builder.rv("log_vol", GaussianRandomWalk, %{sigma: "sigma_h"}, shape: {n})

logpdf_sv = fn _x, params ->
  obs = params.__obs_data
  sigma_t = Nx.exp(Nx.divide(params.log_vol, SSM.t(2.0)))
  sigma_t = Nx.max(sigma_t, SSM.t(1.0e-6))
  SSM.normal_logpdf_vec(obs, params.trend, sigma_t)
end

dist = Custom.new(logpdf_sv, support: :real)

ir = Custom.rv(ir, "lik", dist, %{
    trend: "trend", log_vol: "log_vol", __obs_data: "__obs_data"
  })
  |> Builder.obs("lik_obs", "lik", Nx.tensor(0.0, type: :f64))

{trace, _} = Sampler.sample(ir,
  %{"sigma_trend" => 0.3, "sigma_h" => 0.2,
    "trend" => y_tensor,
    "log_vol" => Nx.broadcast(Nx.tensor(0.0, type: :f64), {n})},
  num_warmup: 1000, num_samples: 1000)

# Extract time-varying volatility
vol_path = trace["log_vol"]
  |> Nx.mean(axes: [0])
  |> Nx.divide(2)
  |> Nx.exp()
  |> Nx.to_flat_list()
```

**What you get**: `vol_path` shows when the process was calm and when it was
turbulent — without being told when the crises were.

## Model 3: Trend + Seasonal

Decompose a series into trend and repeating seasonal pattern.

```
trend_{t+1} = trend_t + η_t
seasonal_t = -sum(seasonal_{t-1}, ..., seasonal_{t-s+1}) + ω_t
y_t = trend_t + seasonal_t + ε_t
```

For period `s` (e.g., s=12 for monthly data with yearly cycle), the seasonal
component sums to approximately zero over each complete cycle.

```elixir
s = 12  # seasonal period

ir = Builder.new_ir()
  |> Builder.data(y)
  |> Builder.rv("sigma_trend", HalfNormal, %{sigma: SSM.t(1.0)})
  |> Builder.rv("sigma_season", HalfNormal, %{sigma: SSM.t(0.5)})
  |> Builder.rv("sigma_obs", HalfNormal, %{sigma: SSM.t(1.0)})
  |> Builder.rv("trend", GaussianRandomWalk, %{sigma: "sigma_trend"}, shape: {n})
  |> Builder.rv("seasonal", GaussianRandomWalk, %{sigma: "sigma_season"}, shape: {n})

logpdf_seasonal = fn _x, params ->
  obs = params.__obs_data
  trend = params.trend
  seasonal = params.seasonal

  # Seasonal constraint: penalize if seasonal doesn't sum to ~0 over each period
  # (soft constraint via additional log-density term)
  seasonal_list = Nx.to_flat_list(seasonal)
  constraint = seasonal_list
    |> Enum.chunk_every(s)
    |> Enum.map(fn chunk -> Enum.sum(chunk) end)
    |> Enum.map(fn s -> -0.5 * s * s / 0.01 end)  # tight penalty
    |> Enum.sum()

  mu = Nx.add(trend, seasonal)
  ll = SSM.normal_logpdf_vec(obs, mu, params.sigma_obs)
  Nx.add(ll, SSM.t(constraint))
end
```

## Model 4: Autoregressive (AR) via Custom Logpdf

AR(1): `x_t = phi * x_{t-1} + η_t`. The state depends on its previous value
with a coefficient `phi`.

```elixir
ir = Builder.new_ir()
  |> Builder.data(y)
  |> Builder.rv("phi", Normal, %{mu: SSM.t(0.0), sigma: SSM.t(1.0)})
  |> Builder.rv("sigma", HalfNormal, %{sigma: SSM.t(1.0)})

logpdf_ar1 = fn _x, params ->
  obs = params.__obs_data
  phi = params.phi
  sigma = Nx.max(params.sigma, SSM.t(1.0e-6))
  n = Nx.axis_size(obs, 0)

  # AR(1): y_t | y_{t-1} ~ Normal(phi * y_{t-1}, sigma)
  y_prev = obs[0..(n-2)]
  y_curr = obs[1..(n-1)]
  mu = Nx.multiply(phi, y_prev)
  SSM.normal_logpdf_vec(y_curr, mu, sigma)
end

dist = Custom.new(logpdf_ar1, support: :real)

ir = Custom.rv(ir, "lik", dist, %{
    phi: "phi", sigma: "sigma", __obs_data: "__obs_data"
  })
  |> Builder.obs("lik_obs", "lik", Nx.tensor(0.0, type: :f64))

{trace, _} = Sampler.sample(ir,
  %{"phi" => 0.5, "sigma" => 1.0},
  num_warmup: 500, num_samples: 500)

# Posterior of AR coefficient
phi_mean = Nx.mean(trace["phi"]) |> Nx.to_number()
# phi > 0: persistent, phi < 0: oscillating, |phi| > 1: explosive
```

## Model 5: Regime-Switching (Hidden Markov-like)

The trading system's model: multiple regimes, each with different dynamics.

```elixir
# See lib/exmc/trading/regime_model.ex for the production version.
# Simplified:

logpdf_regime = fn _x, params ->
  obs = params.__obs_data

  # Three regimes: trending, mean-reverting, volatile
  mu_trend = params.mu_trend
  sigma_trend = Nx.max(params.sigma_trend, SSM.t(1.0e-8))
  sigma_mr = Nx.max(params.sigma_mr, SSM.t(1.0e-8))
  sigma_vol = Nx.max(params.sigma_vol, SSM.t(1.0e-8))

  # Regime weights (softmax of logit parameters)
  ew1 = Nx.exp(Nx.min(params.logit_w1, SSM.t(10.0)))
  ew2 = Nx.exp(Nx.min(params.logit_w2, SSM.t(10.0)))
  z = Nx.add(SSM.t(1.0), Nx.add(ew1, ew2))
  w0 = Nx.divide(SSM.t(1.0), z)  # P(volatile)
  w1 = Nx.divide(ew1, z)          # P(trending)
  w2 = Nx.divide(ew2, z)          # P(mean-reverting)

  # Log-likelihood per regime
  ll_trend = normal_logpdf_per_obs(obs, mu_trend, sigma_trend)
  ll_mr = normal_logpdf_per_obs(obs, SSM.t(0.0), sigma_mr)
  ll_vol = normal_logpdf_per_obs(obs, SSM.t(0.0), sigma_vol)

  # Log-sum-exp mixture
  mix = Nx.log(Nx.add(Nx.add(
    Nx.multiply(w0, Nx.exp(ll_vol)),
    Nx.multiply(w1, Nx.exp(ll_trend))),
    Nx.multiply(w2, Nx.exp(ll_mr))))

  Nx.sum(mix)
end
```

## Init Values: The Most Common Mistake

NUTS requires initial values for ALL free random variables, including
vector-valued states. The most common error:

```elixir
# ❌ WRONG: missing "trend"
%{"sigma_trend" => 0.5, "sigma_obs" => 1.0}
#=> ** (KeyError) key "trend" not found

# ✅ RIGHT: include the full state vector
%{"sigma_trend" => 0.5, "sigma_obs" => 1.0,
  "trend" => Nx.tensor(your_data, type: :f64)}
```

**Best practice**: initialize the state vector to the data itself. The sampler
will smooth it from there.

For stochastic volatility:
```elixir
"log_vol" => Nx.broadcast(Nx.tensor(0.0, type: :f64), {n})
# log_vol = 0 everywhere means volatility = exp(0/2) = 1 (unit volatility)
```

## How NUTS Handles the State Vector

People ask: "You're sampling a 200-dimensional vector — isn't that slow?"

NUTS sees the flat parameter vector:
```
[σ_trend, σ_obs, trend_1, trend_2, ..., trend_200]
 └─ 2 scalars ─┘  └──────── 200 latent states ──┘
```

The gradient of the `GaussianRandomWalk` logpdf is **sparse**: each `trend_t`
only depends on `trend_{t-1}` and `trend_{t+1}`. This creates a tridiagonal
Hessian structure. NUTS exploits this implicitly — the leapfrog integrator
follows the local curvature, which only connects neighbors.

Performance on the 88-core server:
- 100-point local level: ~3 seconds
- 200-point stochastic volatility: ~10 seconds
- 120-point trend-cycle (lecture demo): ~8 seconds

## Signal-to-Noise Ratio: The Key Diagnostic

The ratio `σ_trend / σ_obs` controls how smooth the inferred trend is:

| σ_trend / σ_obs | Behavior |
|---|---|
| → 0 | Nearly flat trend (all variation is noise) |
| ~ 0.1 | Smooth trend, lots of noise |
| ~ 1.0 | Trend tracks data closely |
| → ∞ | Trend = data (no smoothing) |

This is the Bayesian equivalent of the HP filter's λ parameter — but estimated
from data, not chosen by convention.

```elixir
snr = Nx.mean(trace["sigma_trend"]) |> Nx.to_number()
    / Nx.mean(trace["sigma_obs"]) |> Nx.to_number()
IO.puts("Signal-to-noise ratio: #{Float.round(snr, 3)}")
```

## Which Model to Choose

```
Is the series clearly nonstationary (trend)?
  YES → Local Level as baseline
    Is volatility visibly changing over time?
      YES → Add Stochastic Volatility
    Is there a repeating pattern?
      YES → Add Seasonal component
  NO → Is there autocorrelation?
    YES → AR(p) model
    NO → Are there regime shifts?
      YES → Regime-switching mixture
      NO → Simple Normal, no time series needed
```

## Existing Notebooks

| Notebook | Model | Data |
|---|---|---|
| `trend_cycle_demo.livemd` | Local level + SV + bivariate | FRED (GDP, unemployment, PCE) |
| `13_bayesian_spc.livemd` | Conjugate + BOCPD + changepoint | Nile River + piston rings |
| `15_bearing_degradation.livemd` | Exponential degradation (state = health) | FEMTO bearings |

## References

- Harvey, A.C. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge University Press.
- Durbin, J. & Koopman, S.J. (2012). *Time Series Analysis by State Space
  Methods*. 2nd ed. Oxford University Press.
- Kim, S., Shephard, N. & Chib, S. (1998). "Stochastic Volatility: Likelihood
  Inference and Comparison with ARCH Models." *Review of Economic Studies*.
- Stock, J. & Watson, M. (2007). "Why Has U.S. Inflation Become Harder to
  Forecast?" *JMCB*.
