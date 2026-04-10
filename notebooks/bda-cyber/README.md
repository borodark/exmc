# Bayesian Data Analysis for Cybersecurity

## The Teaching Philosophy

The first probability textbook I owned was full of artillery problems.
Range estimation, shell dispersion, target acquisition under fog. I loved
it. But probability theory is not about artillery, and the examples kept
most people out.

Every field has this problem. The theory is universal — Bayes' theorem
does not care whether you are estimating a sex ratio, a shell trajectory,
or an IDS true positive rate. But the learner cares. A nurse learns
faster from clinical examples. A pilot learns faster from flight
scenarios. A security analyst learns faster from SOC data. Not because
the math is different — because the **context is already loaded**. You
don't spend cognitive effort translating "what is a placenta previa?" or
"why do I care about windshield hardness?" You recognize the problem
instantly, and all your attention goes to the new idea.

This is the argument for domain-specific editions of foundational texts.
The theory belongs to everyone. The examples should belong to the reader.

With language models capable of good technical prose, there is no longer
an excuse. If you can write a worked example about artillery, you can
write the same example about alert triage, or crop yields, or insurance
claims, or satellite orbits. The math doesn't change. The motivation
does. And motivation is the entire bottleneck in quantitative education.

**This track is a proof of concept.** Every notebook here teaches the
same Bayesian method as the corresponding chapter in Gelman et al.'s
*Bayesian Data Analysis* — but on cybersecurity data. A parallel [BDA
track](../bda/) covers the textbook's original examples. The two tracks
are interchangeable. Read whichever one makes the theory click faster.

If you work in a different field — manufacturing, finance, medicine,
logistics — the same approach applies. Take the theory. Replace the
examples. Publish it. The math will wait for you. The examples should
meet you where you are.

---

The methods from Gelman et al.'s *Bayesian Data Analysis* applied to
security operations — IDS alert triage, incident response, vulnerability
prioritization, and threat modeling. Built in Elixir Livebook using
[eXMC](https://github.com/borodark/exmc).

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
