# Two Thousand Particle Filters You Run Three Times

The O-SMC² endurance benchmark's full-scale test — 400 parameter particles,
200 state particles, 200 time steps, window of 30 — took ninety-two minutes.
Memory was impeccable: 80 megabytes at startup, 91 megabytes at the end, no
leak. The algorithm was correct: posteriors converged, ESS stayed healthy, six
of seven tests passed outright. The seventh, the full-scale run, was marked
SLOW.

Ninety-two minutes for 200 observations is 27.7 seconds per time step. For an
algorithm whose selling point is real-time sequential inference, this is the
kind of number that invites questions about the sales pitch.

## The Smoking Gun

The log showed 121 rejuvenations out of 200 time steps. A 60.5% rejuvenation
rate. Every time the effective sample size of the parameter particles dipped
below half — which was most of the time — the algorithm triggered its most
expensive operation.

Each rejuvenation ran three rounds of particle filter computation:

**Round 1**: For each of 400 resampled θ-particles, run a full windowed
particle filter (200 particles, 30 time steps) to compute the current
log-evidence. This is the denominator of the Metropolis-Hastings ratio.
400 PF runs.

**Round 2**: PMCMC rejuvenation. Each θ-particle gets 3 MH moves. Each move
proposes a new θ*, runs a windowed PF to compute the proposed log-evidence, and
accepts or rejects. 400 × 3 = 1,200 PF runs.

**Round 3**: After rejuvenation, re-run the particle filter on each accepted
θ to reconstruct fresh particle states for the next time step. 400 PF runs.

Per rejuvenation: approximately 2,000 particle filter runs. Each PF propagates
200 particles through 30 time steps: 6,000 state transitions, 6,000
log-likelihood evaluations, 30 stratified resamples. Over 121 rejuvenations:
1.45 billion particle propagations.

## Sprint 1: The Data Structures

Three mechanical problems hiding in plain sight.

**List append.** The observation history accumulated via `obs_hist ++ [y_t]`,
which copies the entire list on every step. O(T) per step, O(T²) cumulative.
At T=200: 20,000 unnecessary copy operations. At T=500 (the memory endurance
test): 125,000. The fix: `[y_t | obs_window]` with `Enum.take` to bound the
list to the window size. Prepend is O(1). The observation history that once
grew without bound now holds exactly 30 elements.

**Quadratic resampling.** Stratified resampling selects N indices and looks up
the corresponding particles. The lookup used `Enum.at(particles, idx)` on a
linked list — O(N) per call. Called N=200 times per resample: O(N²) = 40,000
list traversals. Across millions of resample calls (every PF step, every
θ-particle, every time step), this was not negligible. The fix:
`List.to_tuple(particles)` before the resampling loop, then `elem(ptuple, idx)`
for O(1) access. The resample function went from O(N²) to O(N).

**Unbounded history.** The `obs_hist` list grew to T elements but only the last
`window` observations were ever referenced. At T=200 with window=30, the
algorithm carried 170 observations it never read. Minor in isolation; a sign
of inattention at scale.

## Sprint 2: The Redundant Work

**Round 3 was unnecessary.** PMCMC already ran a particle filter on the
accepted θ — it had the particle states. It just was not returning them. The
rejuvenation function returned `{theta, log_evidence, accepted}` and threw
away the PF result. The calling code then ran 400 fresh particle filters to
reconstruct what PMCMC had just computed and discarded.

The fix: `PMCMC.rejuvenate` returns a four-tuple
`{theta, log_evidence, accepted, pf_state}` where `pf_state` is
`%{particles: [...], log_weights: [...]}` from the last accepted PF run. The
caller uses it directly. Round 3 — 400 particle filter runs per rejuvenation
— deleted.

**Fixed PMCMC moves.** Every θ-particle received exactly three MH moves
regardless of whether the first one accepted. But a particle whose first
proposal is accepted has already been diversified. The second and third moves
provide diminishing returns for particles that are already exploring well.
They are most needed for particles stuck in rejection.

The fix: `Enum.reduce_while` replaces `Enum.reduce`. On acceptance, halt. On
rejection, continue to the next move. With typical acceptance rates of 20–40%,
roughly half of particles stop after one move, cutting PMCMC cost by a third.

This is a pragmatic optimization, not a theoretical nicety. The kernel is no
longer fixed-scan. For applications requiring strict theoretical PMCMC
guarantees, the `adaptive_moves: false` option restores the old behavior.

## The Numbers

|                        | Before        | After         |
|------------------------|---------------|---------------|
| smoke (warm)           | 1,312 ms      | 714 ms        |
| PF runs per rejuv.     | ~2,000        | ~1,000 (est.) |
| Resampling complexity  | O(N²)         | O(N)          |
| Observation history    | O(T) growing  | O(window)     |
| **Speedup**            |               | **1.8×**      |

All seven tests pass. The public API — `SMC.run/4`, `SMC.filter/3` — is
unchanged. The optimizations are entirely internal.

## What Remains

Sprint 3 targets the remaining waste:

**Adaptive ESS threshold.** The current fixed threshold (0.5 × Nθ) triggers
rejuvenation 60.5% of the time. A two-tier scheme — soft threshold at 0.5
with one PMCMC move, hard threshold at 0.3 with full moves — could reduce the
rejuvenation rate to ~35%. Projected: 36% reduction in rejuvenation cost.

**Incremental evidence caching.** Round 1 computes the windowed log-evidence
from scratch for each θ-particle. But the incremental log-likelihood from each
BPF step is already computed during normal operation. Accumulate the last
`window` values in a rolling buffer: Round 1 eliminated entirely.

Together with Sprints 1 and 2, the projected total speedup is 3.8× — from
92 minutes to approximately 24 minutes for the full-scale benchmark.

## The Lesson

None of these fixes changed what the algorithm computes. The posterior is the
same posterior. The particle filter is the same particle filter. What changed
is how often we ran it, whether we kept its output, and what data structure
held the particles between runs.

The most expensive bugs are the ones that produce correct output slowly. They
survive code review because the tests pass. They survive benchmarking at small
scale because O(N²) feels like O(N) when N is 50. They surface only when
someone runs the full endurance suite and asks why 200 observations take an
hour and a half.

The particle filter code was algorithmically correct from the start. The
waste was not in the algorithm. It was in the plumbing.

## P.S.

The full endurance suite finished while this was being written. Sprint 3 was
not needed.

|                             | Before           | After             | Factor    |
|-----------------------------|------------------|-------------------|-----------|
| smoke (T=40)                | 1,312 ms         | 676 ms            | 1.9×      |
| medium (T=100)              | 5,843 ms         | 3,764 ms          | 1.6×      |
| time-varying-beta (T=120)   | 136,067 ms       | 13,194 ms         | **10.3×** |
| high-rejuv (T=80)           | 25,188 ms        | 4,598 ms          | **5.5×**  |
| full-scale (Nθ=400, T=200)  | 5,550,413 ms     | 317,234 ms        | **17.5×** |
| memory-T500                 | 96,877 ms        | 15,886 ms         | **6.1×**  |
| **Total**                   | **5,813 s**      | **356 s**          | **16.3×** |
| Result                      | 6/7 OK           | **7/7 OK**         |           |

The full-scale test went from ninety-two minutes to five minutes and eighteen
seconds. The projection was 3.8×. The measurement was 17.5×.

The time-varying-beta test — the one with the highest rejuvenation frequency —
improved by 10.3×. This is exactly where eliminating Round 3 pays off: the
more often you rejuvenate, the more particle filter runs you were throwing
away and recomputing. That test rejuvenated 56 out of 120 time steps. At 400
PF runs per redundant round, that is 22,400 particle filter runs deleted.

The previously-SLOW full-scale test now passes. Seven of seven. The suite that
took ninety-seven minutes runs in under six.
