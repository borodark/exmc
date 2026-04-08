# Bayesian Data Analysis in Elixir Livebooks

A port of Aki Vehtari's [BDA Python demos](https://github.com/avehtari/BDA_py_demos) for the textbook **Bayesian Data Analysis, 3rd ed.** (Gelman, Carlin, Stern, Dunson, Vehtari, Rubin) into Elixir Livebooks using the eXMC probabilistic programming framework.

## Why This Exists

BDA3 is the standard graduate text for applied Bayesian statistics. Vehtari's
demos are the canonical teaching material — used in his Aalto University
course and translated into Matlab, R, and Python. This port brings the same
pedagogy to Elixir/Livebook with two contributions on top of replication:

1. **Same examples, native runtime.** Every model that Vehtari teaches with
   PyMC/Stan/scipy is built directly with `Exmc.Builder`. The reader compares
   the math in BDA3 to the IR they construct.
2. **Educator spirit, not transcription.** Each notebook follows the
   established eXMC notebook template (see `notebooks/12_poker_bayesian.livemd`):
   motivating prose, mathematical formulation, study guide with exercises,
   literature pointers, cross-references to other eXMC notebooks.

## Source Inventory (22 demos, 8 chapters)

| BDA3 Ch | Demo | Topic | Datasets | Methods |
|---|---|---|---|---|
| 2 | demo2_1 | Beta posterior, placenta previa | (438, 544) | Analytical |
| 2 | demo2_2 | Prior sensitivity | (438, 544) | Analytical |
| 2 | demo2_3 | Sampling + transformation | (438, 544) | Monte Carlo |
| 2 | demo2_4 | Grid posterior, non-conjugate prior | (438, 544) | Grid + inverse-CDF |
| 3 | demo3_1-4 | Normal model μ,σ² | windshieldy | Normal-inverse-χ² |
| 3 | demo3_5 | Newcomb light-speed | light.txt | Normal posterior + outliers |
| 3 | demo3_6 | Bioassay grid posterior | (x,n,y) inline | 2D grid + sampling |
| 4 | demo4_1 | Laplace approximation for bioassay | (x,n,y) inline | grad/Hessian |
| 5 | demo5_1 | Rats experiment | inline | Hierarchical Beta-binomial |
| 5 | demo5_2 | SAT 8-schools | y,s inline | Hierarchical Normal |
| 6 | demo6_1 | Posterior predictive | placenta | PPC |
| 6 | demo6_2 | Sequential dependence test | inline | PPC test statistics |
| 6 | demo6_3 | Light speed, bad test stat | light.txt | PPC pitfall |
| 6 | demo6_4 | Light speed, good test stat | light.txt | PPC done right |
| 9 | demo9_1 | Jar of coins decision | N(160,40) prior | Bayesian decision theory |
| 10 | demo10_1 | Rejection sampling | toy | Algorithm |
| 10 | demo10_2 | Importance sampling | toy | Algorithm |
| 10 | demo10_3 | Normal approximation, bioassay | (x,n,y) | Multivariate Laplace |
| 11 | demo11_1 | Gibbs sampling | bivariate normal | Algorithm |
| 11 | demo11_2 | Metropolis sampling | bivariate normal | Algorithm |
| cmdstanpy | bern.stan | Bernoulli | algae | Stan example |
| cmdstanpy | binom.stan, binomb.stan, binom2.stan | Binomial variants | algae | Stan example |
| cmdstanpy | lin.stan, lin_std.stan, lin_t.stan | Linear regression | kilpisjarvi | Stan example |
| cmdstanpy | logistic_t.stan, logistic_hs.stan | Logistic regression | diabetes | Sparsity priors |
| cmdstanpy | grp_aov.stan, grp_prior_mean*.stan | Hierarchical group means | factory | One-way ANOVA |

## Vendored Data

Copied to `notebooks/bda/data/`:

| File | Source | Used in |
|---|---|---|
| `algae.txt` | BDA Aalto course | bern.stan, binom.stan demos |
| `drowning.txt` | BDA Aalto course | linear trend example |
| `factory.txt` | BDA Aalto course (5×6 quality measurements) | grp_*.stan |
| `windshieldy1.txt`, `windshieldy2.txt` | BDA3 Ch 3 | normal model |
| `light.txt` | Newcomb 1882 (BDA3 p. 66) | normal model + PPC |
| `kilpisjarvi-summer-temp.csv` | Finnish Met. Inst. | linear regression |
| `diabetes.csv` | Pima Indians | logistic regression |

## eXMC Capability Mapping

| BDA construct | eXMC equivalent | Status |
|---|---|---|
| Beta(α, β) prior | `Exmc.Dist.Beta` | ✅ |
| Normal(μ, σ) | `Exmc.Dist.Normal` | ✅ |
| Half-Normal | `Exmc.Dist.HalfNormal` | ✅ |
| Cauchy / Half-Cauchy | `Exmc.Dist.Cauchy`, `Exmc.Dist.HalfCauchy` | ✅ |
| Student-t (robust regression) | `Exmc.Dist.StudentT` | ✅ |
| Bernoulli / Binomial | `Exmc.Dist.Bernoulli` (binomial via vectorized obs) | ✅ |
| Poisson | `Exmc.Dist.Poisson` | ✅ |
| Multivariate Normal | `Exmc.Dist.MvNormal` | ✅ |
| Dirichlet | `Exmc.Dist.Dirichlet` | ✅ |
| Truncated / censored | `Exmc.Dist.TruncatedNormal`, `Exmc.Dist.Censored` | ✅ |
| Mixture | `Exmc.Dist.Mixture` | ✅ |
| Hierarchical via string refs | `Builder.rv(ir, "alpha", N, %{mu: "mu_pop", sigma: "sigma_pop"})` | ✅ |
| Non-centered parameterization | Automatic via `ncp: true` rewrite (or `ncp: false` to disable) | ✅ |
| NUTS sampling | `Exmc.Sampler.sample/3` | ✅ |
| Multi-chain parallel | `Sampler.sample_chains/4` | ✅ |
| Posterior predictive | `Exmc.Predictive` | ✅ |
| WAIC / LOO | `Exmc.Comparison` | ✅ |
| Trace plots, R-hat, ESS | `ExmcViz` (Scenic) or VegaLite cells | ✅ |
| **Grid posterior** | Hand-rolled with `Nx.linspace` + broadcasting | Educational |
| **Rejection sampling** | Hand-rolled, ~20 lines of Elixir | Educational |
| **Importance sampling** | Hand-rolled, ~20 lines | Educational |
| **Gibbs sampling** | Hand-rolled, ~30 lines | Educational |
| **Metropolis sampling** | Hand-rolled, ~30 lines | Educational |
| **Laplace approximation** | `Nx.Defn.grad` + `Nx.LinAlg.invert` (~20 lines) | Educational |

The "educational" rows are *not* missing features. BDA3 chapters 10-11 *teach
students how to implement these from scratch*. Hand-rolling them in Elixir is
the point of those chapters, not a workaround.

## Notebook Plan

```
notebooks/bda/
├── PLAN.md                              ← this file
├── data/                                ← vendored datasets (8 files)
├── ch02_beta_binomial.livemd            ← demos 2_1..2_4 merged (placenta previa)
├── ch03_normal_and_bioassay.livemd      ← demos 3_1..3_6 (windshieldy + Newcomb + bioassay)
├── ch04_normal_approximation.livemd     ← demo 4_1 (Laplace for bioassay)
├── ch05_eight_schools.livemd            ← demos 5_1, 5_2 (rats + 8-schools, NUTS)
├── ch06_posterior_predictive.livemd     ← demos 6_1..6_4
├── ch09_decision_analysis.livemd        ← demo 9_1
├── ch10_rejection_importance.livemd     ← demos 10_1, 10_2 (algorithms from scratch)
├── ch11_gibbs_metropolis.livemd         ← demos 11_1, 11_2 (algorithms from scratch)
└── stan_translations.livemd             ← all 13 .stan files as eXMC IR side-by-side
```

## Educator Template (every notebook)

The `12_poker_bayesian.livemd` pattern, applied uniformly:

```markdown
# <Chapter title>

## Setup
   <CPU-only Mix.install boilerplate>

## Why This Matters
   <Concrete motivation. The reader should care before they see math.>

## The Problem
   <Dataset description, BDA3 page reference, what we're trying to learn.>

## The Model
   <Mathematical formulation with LaTeX, then eXMC IR build,
    then a sentence-by-sentence walkthrough.>

## Sampling / Computing
   <Either Sampler.sample, or hand-rolled algorithm. With diagnostics.>

## Visualization
   <VegaLite plots — posterior, trace, predictive checks.>

## What This Tells You
   <Interpretation tied back to the original question.>

## Study Guide
   <3-5 exercises matching BDA3 end-of-chapter problems.
    Each exercise references a specific cell to modify.>

## Literature
   - BDA3 §X.Y, p. NNN (the canonical reference)
   - Vehtari et al. paper (when relevant — PSIS, R-hat, etc.)
   - Cross-link to other eXMC notebooks that build on this material
```

## Pacing

Two pilots first, in order of educational dependency:

1. **Ch 2 — Beta-binomial.** Pure analytical. No sampler. Tests the educator
   template against foundational material. Proves the BDA3 math translates
   cleanly to Nx + VegaLite.

2. **Ch 5 — 8-schools.** The canonical hierarchical model. Showcases NUTS,
   NCP, multi-chain parallel sampling. The "hello world" of MCMC
   convergence diagnostics. Stress-tests eXMC's sampler against PyMC's
   reference results.

After both pilots are validated and verified (study guide checked,
literature linked, plots match BDA3 figures), the remaining 6 notebooks can
be ported in any order — they don't depend on each other.

## Cross-References to Existing eXMC Notebooks

| BDA Ch | Builds toward | Existing eXMC notebook |
|---|---|---|
| 2 | `01_getting_started.livemd` | the next step after binomial |
| 3 (bioassay) | `12_poker_bayesian.livemd` | logistic + softmax for behavior |
| 4 (Laplace) | `04_variational_inference.livemd` | the next-fancier approximation |
| 5 (8-schools) | `02_hierarchical_model.livemd`, `09_radon_bhm.livemd` | partial pooling at scale |
| 6 (PPC) | `13_bayesian_spc.livemd` | predictive checks in production |
| 9 (decision) | `06_dca_business.livemd` | decision under uncertainty |
| 10-11 (algorithms) | every notebook that uses NUTS | "what's the sampler doing?" |
| Stan grp_aov | `02_hierarchical_model.livemd` | one-way random effects |

## License & Attribution

The original demos are by Aki Vehtari (Aalto), Tuomas Sivula, Pellervo Ruponen,
Lassi Meronen, Osvaldo Martin, and contributors, under the BSD-3-Clause license.
This Elixir port preserves attribution and references each source demo by
filename. Datasets are reproduced under the original BDA3 educational use terms.

## Status

- [x] Survey complete (this document)
- [x] Ch 2 pilot — `ch02_beta_binomial.livemd` (placenta previa, analytical Beta + grid)
- [x] Ch 5 pilot — `ch05_eight_schools.livemd` (NUTS hierarchical, NCP vs centered A/B)
- [x] Ch 3 — `ch03_normal_and_bioassay.livemd` (windshieldy, Newcomb, bioassay grid)
- [x] Ch 4 — `ch04_normal_approximation.livemd` (Laplace via Newton + finite-diff Hessian)
- [x] Ch 6 — `ch06_posterior_predictive.livemd` (PPC done well and badly)
- [x] Ch 9 — `ch09_decision_analysis.livemd` (jar of coins, expected utility)
- [x] Ch 10 — `ch10_rejection_importance.livemd` (hand-rolled rejection + IS)
- [x] Ch 11 — `ch11_gibbs_metropolis.livemd` (hand-rolled Gibbs + Metropolis)
- [x] Stan companion — `stan_translations.livemd` (13 .stan files side-by-side)
- [ ] Verification via livebook-verify skill (Ch 2 + Ch 3 + Ch 4 + Ch 10 + Ch 11 numerics confirmed via mix run)
- [ ] Persisted outputs in Livebook (manual step before publication)
