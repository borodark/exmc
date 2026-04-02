# Changelog

## 0.2.0 (2026-03-30)

- Warm-start NUTS: reuse previous mass matrix + step size (5.8x speedup)
- 21 distributions (Lognormal, HalfCauchy, TruncatedNormal, Bernoulli, Poisson added)
- Builder.data/2 API for JIT-safe observation data (fixes 256GB memory leak)
- 4 new notebooks (Bayesian SPC, Bearing Degradation, Turbofan Fleet, State-Space)
- 4 new docs (Warm Start, State Space Models, Scheduler Pinning, Forest Tracker)
- Les Trois Chambrées cross-references (smc_ex, StochTree-Ex)
- Beats PyMC on 4 of 7 benchmarks (medium 1.90x, stress 1.16x, eight_schools 2.55x, sv 1.20x)

## 0.1.0 (2026-01-15)

Initial release.

- NUTS sampler with Stan-style three-phase warmup
- ADVI (mean-field variational inference)
- SMC (likelihood tempering)
- Pathfinder (L-BFGS initialization)
- 16 distributions with automatic constraint transforms
- Streaming inference via sample_stream/4
- Distributed MCMC across Erlang nodes
- 337 tests, 33/33 posteriordb validation
