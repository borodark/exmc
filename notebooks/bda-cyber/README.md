# Bayesian Data Analysis for Cybersecurity

The methods from Gelman et al.'s *Bayesian Data Analysis* applied to
security operations — IDS alert triage, incident response, vulnerability
prioritization, and threat modeling. Built in Elixir Livebook using
[eXMC](https://github.com/borodark/exmc).

A parallel track to the [BDA notebooks](../bda/). Same math. Same framework.
Different domain.

## The Argument

Every security tool gives you a score. None of them tell you how confident
that score is.

- Your IDS says "high severity." Is the model 51% sure or 99% sure?
- Your CVSS score says 7.8. But is that CVE exploitable *in your config*?
- Your SOC investigated 200 alerts and found 43 real. Is the true positive
  rate 21% ± 3% or 21% ± 15%?

Bayesian methods give you both the estimate and the uncertainty — calibrated,
updatable, and honest about what the data can and cannot tell you.

## Notebooks

| Notebook | What You Learn |
|---|---|
| [ch02 — IDS Rule Effectiveness](ch02_ids_rule_effectiveness.livemd) | Beta-Binomial for TP rates, base rate neglect (why "95% accurate" means 1.9% true positives), comparing and triaging rules |
| [ch03 — Network Baseline & Brute Force](ch03_network_baseline_bruteforce.livemd) | Normal model on DNS query lengths with DGA outliers, logistic dose-response for brute force escalation |
| [ch04 — Laplace Approximation](ch04_laplace_bruteforce.livemd) | Fast approximate posteriors for real-time alert scoring |
| [ch05 — Eight SOCs](ch05_eight_socs.livemd) | Hierarchical incident rates across branch offices — partial pooling for sparse data |
| [ch06 — Threat Model Checking](ch06_threat_model_ppc.livemd) | Posterior predictive checks: is your Poisson threat model consistent with bursty reality? |
| [ch09 — Incident Response](ch09_incident_response.livemd) | Bayesian decision theory: contain ($50K) or risk a missed breach ($2M) at 15% posterior? |
| [ch10 — Anomaly Sampling](ch10_anomaly_sampling.livemd) | Rejection + importance sampling on network traffic mixtures |
| [ch11 — MCMC on Traffic](ch11_mcmc_traffic.livemd) | Gibbs + Metropolis on correlated flow features (bytes vs duration) |
| [Stan Cyber Catalog](stan_cyber_translations.livemd) | Six security models (IDS, CVE rate, AV detection, hierarchical SOC, brute force, robust baseline) as Stan ↔ eXMC side-by-side |

## Who This Is For

Security practitioners — SOC analysts, incident responders, security
engineers, CISOs — who want to move beyond point estimates and binary alerts
toward principled uncertainty quantification. No prior Bayesian experience
required. Each notebook builds from the ground up.

If you have read BDA3 or taken a Bayesian statistics course, the
[BDA track](../bda/) covers the same methods on the textbook's original
examples. This track applies them to problems you face at work.

## Running

Open any `.livemd` file in [Livebook](https://livebook.dev). Each notebook is
self-contained. No GPU required.

```bash
livebook server notebooks/bda-cyber/
```

## Attribution

Bayesian methods from Gelman, Carlin, Stern, Dunson, Vehtari, & Rubin,
*Bayesian Data Analysis*, 3rd ed. (2013). Educator template adapted from
Vehtari's [BDA Python demos](https://github.com/avehtari/BDA_py_demos)
(BSD-3-Clause). Security datasets from public sources (NVD, Verizon DBIR,
LANL, CTU-13). See [PLAN.md](PLAN.md) for full dataset provenance.
