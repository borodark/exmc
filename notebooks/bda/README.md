# Bayesian Data Analysis in Elixir Livebook

Aki Vehtari's [BDA Python demos](https://github.com/avehtari/BDA_py_demos) for **Bayesian Data Analysis, 3rd ed.** (Gelman, Carlin, Stern, Dunson, Vehtari, Rubin), translated into Elixir Livebook using the [eXMC](https://github.com/borodark/exmc) probabilistic programming framework.

Same examples. Same pedagogy. Different runtime.

## What This Is

BDA3 is the standard graduate text for applied Bayesian statistics. Vehtari's demos are the canonical teaching material — used at Aalto University and translated into R, Python, and Matlab. This is the Elixir translation.

Every model that Vehtari teaches with PyMC, Stan, or scipy is built here with `Exmc.Builder`. The algorithms taught from scratch (grid posteriors, rejection sampling, importance sampling, Gibbs, Metropolis) are implemented from scratch in Elixir. The reader compares the math in BDA3 to the code the runtime executes.

## Notebooks

| Notebook | BDA3 Chapter | What You Learn |
|---|---|---|
| [ch02_beta_binomial](ch02_beta_binomial.livemd) | Ch 2 | Conjugate Beta–Binomial, prior sensitivity, Monte Carlo sampling, non-conjugate grid posterior |
| [ch03_normal_and_bioassay](ch03_normal_and_bioassay.livemd) | Ch 3 | Normal model with unknown mean/variance, Newcomb's light-speed outliers, 2D bioassay grid posterior |
| [ch04_normal_approximation](ch04_normal_approximation.livemd) | Ch 4 | Laplace approximation via Newton's method and finite-difference Hessian |
| [ch05_eight_schools](ch05_eight_schools.livemd) | Ch 5 | Hierarchical models, partial pooling, the funnel, NUTS, centered vs non-centered parameterization |
| [ch06_posterior_predictive](ch06_posterior_predictive.livemd) | Ch 6 | Posterior predictive checks — how to detect model misfit, and how to pick the right test statistic |
| [ch09_decision_analysis](ch09_decision_analysis.livemd) | Ch 9 | Bayesian decision theory — the jar of coins, expected utility, why posteriors are not decisions |
| [ch10_rejection_importance](ch10_rejection_importance.livemd) | Ch 10 | Rejection and importance sampling from scratch — why they fail in high dimensions |
| [ch11_gibbs_metropolis](ch11_gibbs_metropolis.livemd) | Ch 11 | Gibbs and Metropolis from scratch — the ancestors of every modern sampler |
| [stan_translations](stan_translations.livemd) | Stan companion | 13 Stan model files translated side-by-side into eXMC Builder IR |

## What Changes on the BEAM

The models are identical. The math is identical. What changes:

- **Chains are processes, not threads.** `Sampler.sample_chains/4` dispatches across all cores via `Task.async_stream`. No `multiprocessing.Pool`, no GIL, no serialization.
- **Diagnostics stream as messages.** `sample_stream/4` sends each posterior sample to any BEAM process — a Scenic visualization, a Phoenix LiveView, a GenServer computing running R-hat.
- **Failures are isolated.** A chain that hits a numerical singularity crashes its process. The supervisor restarts it. The other chains keep running.
- **Nx tensors compile to XLA.** `Exmc.JIT` dispatches gradient computation to EXLA (CPU or CUDA GPU) and falls back to BinaryBackend when EXLA is unavailable.

## Running

Open any `.livemd` file in [Livebook](https://livebook.dev). Each notebook is self-contained — `Mix.install` handles dependencies. No GPU required; all notebooks run on CPU.

```bash
livebook server notebooks/bda/
```

## Attribution

The original Python demos are by Aki Vehtari, Tuomas Sivula, Pellervo Ruponen, Lassi Meronen, Osvaldo Martin, and contributors, released under the [BSD-3-Clause license](https://github.com/avehtari/BDA_py_demos/blob/master/LICENSE). Datasets are from BDA3 and public sources (Newcomb 1882, Finnish Meteorological Institute, Rubin 1981).

This translation preserves attribution and references each source demo by filename. The Elixir code and prose are original.

**Gelman, A., Carlin, J.B., Stern, H.S., Dunson, D.B., Vehtari, A., & Rubin, D.B.** (2013). *Bayesian Data Analysis*, 3rd ed. Chapman & Hall/CRC.
