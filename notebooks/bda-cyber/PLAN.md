# Bayesian Data Analysis for Cybersecurity — Livebook Track

A cybersecurity-focused parallel of the [BDA notebooks](../bda/), using the same
Bayesian methods from Gelman et al. on security data. Same math, same eXMC
framework, different domain.

## Who This Is For

Security practitioners who want to move beyond point-estimate tools (CVSS
scores, binary IDS alerts, ad-hoc risk ratings) toward principled uncertainty
quantification. No prior Bayesian experience required — each notebook builds
from the ground up.

## The Central Argument

Every security tool gives you a score. None of them tell you how confident
that score is. Bayesian methods give you both — and in security, knowing what
you don't know is more important than knowing what you do.

## Notebook Plan

```
notebooks/bda-cyber/
├── PLAN.md                                ← this file
├── data/                                  ← vendored datasets (<6MB total)
├── ch02_ids_rule_effectiveness.livemd     ← Beta-Binomial: IDS true positive rates
├── ch03_network_baseline_bruteforce.livemd ← Normal model on traffic + logistic brute force
├── ch04_laplace_bruteforce.livemd         ← Laplace approximation for the Ch 3 logistic
├── ch05_eight_socs.livemd                 ← Hierarchical: incident rates across offices/industries
├── ch06_threat_model_ppc.livemd           ← PPC: is our threat model consistent with observed attacks?
├── ch09_incident_response.livemd          ← Decision theory: contain, investigate, or ignore?
├── ch10_anomaly_sampling.livemd           ← Rejection + importance sampling on traffic distributions
├── ch11_mcmc_traffic.livemd               ← Gibbs + Metropolis on correlated network features
└── stan_cyber_translations.livemd         ← Security models side-by-side: Poisson CVE, Beta AV, hierarchical
```

## Chapter-to-Problem Mapping

| Ch | BDA3 Original | Cybersecurity Problem | Key Insight |
|---|---|---|---|
| 2 | Placenta previa sex ratio | IDS rule TP rate with base rate neglect | A "95% accurate" rule produces 1.9% true positives when attacks are 0.1% of traffic |
| 3 | Windshield hardness + Newcomb + bioassay | DNS query length baseline (DGA outliers) + brute force dose-response | Normal model breaks on adversarial outliers; logistic models attack escalation |
| 4 | Laplace for bioassay | Same logistic, approximated | Fast approximate posteriors for real-time scoring |
| 5 | 8 schools SAT coaching | 8 SOCs / 15 industries — incident rates with partial pooling | Small offices borrow strength from large ones; sparse data doesn't mean no information |
| 6 | Newcomb PPC — good vs bad test statistics | Poisson threat model vs bursty reality | A Poisson model for weekly CVEs misses clustering; PPC detects this |
| 9 | Jar of coins | IR triage: contain ($50K) vs miss breach ($2M) at 15% posterior | Expected-loss framing changes every triage decision |
| 10 | Rejection + importance sampling | Sampling from traffic mixture (benign + C2 + scan) | Why these methods fail on high-dimensional flow features |
| 11 | Gibbs + Metropolis on bivariate normal | MCMC on correlated network features (bytes vs duration) | Correlation structure in attack traffic |
| Stan | 13 .stan files | Poisson CVE discovery rate, Beta-Binomial AV detection, hierarchical incidents | Security-specific model catalog |

## Vendored Datasets

| File | Source | Used In | Size | License |
|---|---|---|---|---|
| `ids_alert_summary.csv` | Derived from NSL-KDD test set | Ch 2 | <5KB | UNB academic |
| `lanl_redteam_auth.csv` | LANL Unified Host/Network | Ch 3 | ~700KB | Gov open data |
| `dga_domains.csv` | Published DGA wordlists + Alexa Top 1M sample | Ch 3 | <500KB | CC0 / public |
| `dbir_industry_incidents.csv` | Hand-transcribed from Verizon DBIR 2024 | Ch 5 | <5KB | Public report |
| `nvd_2023_cve_weekly.csv` | NVD API extract | Ch 6 | <100KB | Public domain |
| `nvd_exploitdb_tte.csv` | NVD + ExploitDB join — time-to-exploit | Ch 6, Stan | <200KB | Public domain + GPL |
| `avtest_detection.csv` | AV-TEST monthly engine results | Stan | <2KB | Published data |
| `ctu13_s10_flows.csv` | CTU-13 Scenario 10 subset | Ch 10-11 | ~1MB | CTU academic |

## Cross-References to BDA Track

Each cybersecurity notebook links to its BDA counterpart:

| Cyber Ch | BDA Ch | Why Read Both |
|---|---|---|
| Ch 2 (IDS rules) | Ch 2 (placenta previa) | Same Beta-Binomial, different stakes |
| Ch 3 (baseline + brute force) | Ch 3 (windshieldy + bioassay) | Same normal + logistic, adversarial outliers |
| Ch 5 (eight SOCs) | Ch 5 (eight schools) | Partial pooling at scale — schools vs security ops |
| Ch 9 (IR triage) | Ch 9 (jar of coins) | Decision theory moves from classroom to incident command |

## Pacing

1. **Ch 2 — IDS Rule Effectiveness.** The hook. Base rate neglect is the
   "aha" that makes security practitioners care about Bayes. Pure analytical,
   no sampler. Inline data.

2. **Ch 5 — Eight SOCs.** The career argument. "My small branch had 3
   incidents — am I secure or just small?" Requires NUTS.

3. **Ch 9 — Incident Response.** The operational payoff. Every IR decision
   is a Bayesian decision problem.

After these three pilots, remaining notebooks can proceed in any order.

## Attribution

Bayesian methods from Gelman et al., *Bayesian Data Analysis*, 3rd ed.
Educator template from Vehtari's BDA Python demos (BSD-3-Clause).
Security domain knowledge from open sources (DBIR, NVD, MITRE ATT&CK).
Datasets are public domain, government open data, or academic-use licensed.
