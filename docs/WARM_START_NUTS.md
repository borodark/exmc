# The Five Hundred Iterations You Run Twice

The trading system samples 102 financial instruments every twenty minutes.
Each instrument runs its own NUTS sampler: 500 warmup iterations to adapt
the mass matrix and step size, then 200 sampling iterations to draw from
the posterior. The posterior feeds a regime-switching model that generates
buy/sell/hold signals. This happens on 44 concurrent CPU workers, around
the clock.

The number that should have bothered us earlier: 500.

Five hundred warmup iterations — for a model that ran twenty minutes ago,
on the same instrument, with nearly identical data. The returns shifted
by one tick. The posterior moved by a fraction of a percent. And we were
spending 500 iterations rediscovering a mass matrix that was, for all
practical purposes, the same one we'd just thrown away.

## What Warmup Actually Does

NUTS warmup has three jobs. The first fifty iterations find a reasonable
step size — the leapfrog integrator's stride length. The next several
hundred iterations estimate the mass matrix — a diagonal scaling that
accounts for the posterior's different variances along different
dimensions. The final fifty iterations fine-tune the step size for the
newly estimated mass matrix.

The mass matrix is the expensive part. It requires accumulating
sufficient statistics (Welford online variance) across hundreds of
gradient evaluations, each of which runs through the EXLA JIT compiler.
For the d=8 regime model, each warmup iteration costs roughly the same
as a sampling iteration — about 4 milliseconds. Five hundred warmup
iterations: two full seconds, burned before drawing a single useful
sample.

The mass matrix from twenty minutes ago was sitting right there, in the
Instrument GenServer's state. Nobody was using it.

## The Observation

The mass matrix captures the posterior geometry: which parameters vary
a lot (large diagonal entries) and which are tightly constrained (small
entries). When the data changes by one tick out of two hundred, the
posterior geometry barely shifts. The mass matrix from the previous run
is not merely a good starting point — it is, within numerical tolerance,
the correct answer.

The step size, similarly, depends on the mass matrix. A well-adapted
step size from twenty minutes ago is still well-adapted, because the
mass matrix it was adapted to is still correct.

The only thing warmup needs to do, when given a previous mass matrix, is
a short fine-tuning pass: fifty iterations of dual averaging to nudge the
step size for whatever minor changes the new data introduced.

## The Fix

One keyword argument:

```elixir
{trace, stats} = Sampler.sample(ir, init,
  num_warmup: 200,
  num_samples: 200,
  warm_start: previous_stats    # ← this
)
```

When `warm_start` is provided — a map containing `step_size` and
`inv_mass_diag` from the previous run — the sampler skips mass matrix
initialization, skips the step-size search, and runs only 50 iterations
of fine-tuning instead of the full 500.

The Instrument GenServer already stores `stats` from the last posterior
update. The change to pass it forward was three lines:

```elixir
defp sample_opts(opts, state \\ nil) do
  base = [num_samples: 200, num_warmup: 200, seed: :rand.uniform(10000), ncp: false]
  if state && state.stats && state.stats[:step_size] do
    Keyword.put(base, :warm_start, state.stats)
  else
    base
  end
end
```

The first sampling round for each instrument — from checkpoint, no
previous stats — runs full warmup. Every subsequent round gets the
warm-start path.

## The Numbers

| | Cold start | Warm start |
|---|---|---|
| Warmup iterations | 500 | 50 |
| Wall time | 1,979 ms | 339 ms |
| Step size | 0.828 | 0.749 |
| **Speedup** | | **5.8x** |

The step sizes converge to neighboring values — 0.83 versus 0.75 —
confirming that the previous mass matrix needed only minor adjustment.

For the full trading system: 102 instruments, each saving 1.6 seconds
per update cycle. Over a market day (twenty update cycles in six and a
half hours): 102 × 20 × 1.6 = 3,264 seconds = **54 minutes** of
sampling time saved per day. On 44 concurrent workers, this translates
to roughly 74 seconds of wall time per cycle freed up — headroom for
more instruments or faster response to market events.

## What the BEAM Made Easy

The warm-start pattern is trivial on the BEAM because the GenServer
state persists between sampling rounds. The previous `stats` map lives
in the Instrument process's heap — no serialization, no Redis, no
filesystem checkpoint. It is simply *there*, in memory, from the last
time the process ran `Sampler.sample`.

In Python, achieving the same thing requires either pickling the mass
matrix to disk between invocations, or maintaining a long-lived process
with mutable state and explicit lifecycle management. In Stan, there is
no mechanism at all — each chain starts from the identity mass matrix,
every time.

The BEAM's contribution is not the algorithm (any sampler could accept a
warm-start parameter) but the **zero-cost persistence**: the GenServer's
state is the warm-start cache, supervised, garbage-collected, and
available without a single line of serialization code.

## The Lesson

Not every warmup is warmup. When the posterior moves slowly — and in a
streaming system with rolling windows, it almost always does — the
previous run's adaptation is next run's initialization. The 500
iterations we were burning were not exploring unknown territory. They
were retracing steps, at four milliseconds each, to arrive at a mass
matrix indistinguishable from the one they started with.

The fix was not clever. It was the absence of waste.
