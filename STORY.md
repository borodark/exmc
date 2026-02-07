# The Amber Trace

*A short account of building a probabilistic programming framework on the wrong virtual machine, in the style of Tracy Kidder*

---

## I. The Wrong Language

The idea arrived the way most dangerous ideas do: as a reasonable question. If PyMC could be rebuilt on a different runtime, one where concurrency was not an afterthought but the fundamental unit of abstraction, what would change? Not the math -- Hamiltonian Monte Carlo does not care what language invokes it. The chain explores the same posterior geometry whether the gradient comes from Theano, JAX, or a tensor library written in Elixir by a single person working late at night.

What would change is everything around the math. The orchestration. The four chains that PyMC runs as four separate operating system processes, each loading the model independently, each claiming its own memory, each trapped behind Python's Global Interpreter Lock like prisoners sharing a telephone. In Python you can parallelize MCMC, but only by pretending that each chain lives on a different computer. The serialization overhead, the process startup time, the four separate JIT compilations of the same model -- all of this is the tax you pay for running probabilistic programs in a language that was designed for scripting, not for systems.

Elixir was designed for systems. Not for numerical computing -- nobody would claim that -- but for the kind of distributed, concurrent, fault-tolerant work that telephone switches and messaging platforms demand. The BEAM virtual machine, Elixir's runtime, can spin up millions of lightweight processes. Four MCMC chains are four processes sharing one compiled model, dispatched across all available cores with a single call to `Task.async_stream`. No serialization. No redundant compilation. No GIL.

The question was whether the rest of the story could be made to work. Whether tensor operations, automatic differentiation, constraint transforms, and a No-U-Turn Sampler could be assembled from Elixir's still-young numerical ecosystem into something that would actually sample from a posterior distribution. Whether the BEAM, a virtual machine built for telecommunications, could be taught to do Bayesian statistics.

---

## II. Thirty-Five Decisions

The project acquired the name Exmc early, an abbreviation that pleased nobody but stuck anyway. The first decision -- recorded in a file called `DECISIONS.md` that would eventually grow to thirty-five entries -- was to use Nx as the numerical backbone. Nx is Elixir's tensor library, analogous to NumPy but younger, smaller, and designed from the start to support multiple hardware backends. On CPU, it provides a pure-Elixir "BinaryBackend" that interprets tensor operations directly. When the EXLA library is available, it compiles those operations to XLA -- Google's accelerated linear algebra compiler -- and fuses them into optimized machine code.

Decision two was the intermediate representation. PyMC builds models as Theano (now PyTensor) graphs. Exmc would build models as a small IR: random variable nodes, observation nodes, deterministic nodes, connected by named references. A model is a data structure before it is a computation. You declare the model, rewrite it, compile it, then sample from it. Four layers, each a clean boundary.

The builder API fell out naturally from Elixir's pipe operator:

```
Builder.new_ir()
|> Builder.rv("mu", Normal, %{mu: 0.0, sigma: 5.0})
|> Builder.rv("sigma", HalfNormal, %{sigma: 2.0})
|> Builder.rv("x", Normal, %{mu: "mu", sigma: "sigma"})
|> Builder.obs("x_obs", "x", Nx.tensor([2.1, 1.8, 2.5]))
```

Each pipe adds a node to the IR. String values in parameter maps -- `"mu"`, `"sigma"` -- are references to other random variables, resolved at evaluation time. When the compiler walks this IR, it produces a closure that maps a flat vector of unconstrained values to a log-probability and its gradient. That closure is the heart of the sampler.

The decisions accumulated. Decision four: distributions declare their own constraint transforms. A `HalfNormal` lives on the positive reals, so it declares `:log` as its default transform. The sampler operates in unconstrained space; the transform and its Jacobian correction are folded into the log-probability automatically. Decision ten: the flat vector stores unconstrained values. Decision fifteen: the sampler returns constrained values. The user never sees the unconstrained space unless they go looking for it.

Each decision was a bet. Some of them were wrong. Many of them would need to be revisited, clarified, or partially superseded by later discoveries. But the document kept growing, and the document was honest.

---

## III. The Thousand-Fold Slowdown

Decision sixteen nearly killed the project.

The NUTS sampler -- the No-U-Turn Sampler, the modern standard for Hamiltonian Monte Carlo -- requires random decisions at every step. Which direction to extend the trajectory. How to weight the proposal. Whether to accept or reject. Each decision consumes one or more random draws.

The natural approach in Nx would be to use `Nx.Random`, the library's built-in PRNG. Generate a key, split it, draw from it. This is how JAX handles randomness: pure functional, deterministic, elegant.

On BinaryBackend, each call to `Nx.Random.split` took approximately one second.

Not one millisecond. One second. The operation triggered Nx's `defn` tracing machinery -- the system that analyzes computation graphs for JIT compilation -- but without any JIT compiler to emit. The tracing happened, discovered there was no backend to optimize for, and ran the operation through the interpreter. Every call. For an operation that should take microseconds, the overhead was three to four orders of magnitude.

A single NUTS step might need five or ten random draws. Three hundred warmup iterations plus three hundred sampling iterations meant thousands of seconds -- hours -- for a model with one free parameter.

The fix was decision sixteen: abandon `Nx.Random` entirely and use Erlang's built-in `:rand` module. Erlang's PRNG is a plain scalar function. You give it a state, it gives you a number and a new state. No tracing, no graph analysis, no compilation overhead. The state threads through the sampler explicitly -- a natural fit for functional programming.

```elixir
rng = :rand.seed_s(:exsss, seed)
{value, rng} = :rand.normal_s(rng)
```

Each call took microseconds. The sampler went from hours to seconds. The three-to-four-order-of-magnitude penalty disappeared entirely, replaced by the natural speed of the BEAM's native arithmetic.

This was the first of many encounters with the gap between how a numerical library is supposed to work and how it actually works when you run it on the wrong backend. Nx was designed for EXLA. BinaryBackend was a development convenience, a fallback, a way to run tests without installing a compiler toolchain. Using it as a production backend was like driving a car designed for the autobahn through a plowed field. Everything worked. Nothing was fast.

---

## IV. The Gradient Problem

The sampler came together in pieces. The leapfrog integrator, which steps through Hamiltonian dynamics by alternating momentum and position updates. The tree builder, which extends the trajectory in both directions until it starts to turn back on itself. The mass matrix adapter, which learns the posterior's covariance structure during warmup using Welford's online algorithm. The dual averaging adapter, which tunes the step size to hit a target acceptance rate of 80%.

Three-phase warmup, following Stan's proven schedule: an initial buffer for step-size-only adaptation, a middle section with doubling windows for joint step-size and mass-matrix adaptation, and a terminal buffer to lock in the final step size with frozen mass matrix. The schedule was copied from Stan because Stan got it right and there was no reason to reinvent it.

The first model that sampled correctly was a standard Normal. Mean approximately zero, variance approximately one, 500 samples after 500 warmup iterations. It took about ten seconds on BinaryBackend. The numbers were right.

Then the variance was wrong.

Not dramatically wrong -- 1.5 to 2.0 instead of 1.0, consistently across seeds. The kind of systematic bias that indicates a bug in the dynamics, not in the random number generator. A long debugging session revealed the problem deep in the tree builder: when extending a subtree from a trajectory endpoint, the code was using the gradient at the *proposal* position instead of the gradient at the *endpoint*. In NUTS, the trajectory is a binary tree, and each leaf needs its own gradient. Confuse one leaf's gradient with another's and the dynamics are subtly wrong -- not wrong enough to crash, but wrong enough to inflate the variance by fifty percent.

The fix required threading `grad_left` and `grad_right` fields through every trajectory merge operation. After the fix, the variance came back: 1.066 with 500 samples, which is within the expected range for a finite sample from a unit Normal.

But the gradient problem was not finished. It was only beginning.

---

## V. The lgamma Wall

Exmc supported nine distributions: Normal, HalfNormal, Exponential, Gamma, Beta, Uniform, StudentT, Cauchy, and LogNormal. The first three have simple log-probability functions -- a quadratic here, a linear term there, nothing that threatens a numerical library. Gamma and Beta are different. Their log-probabilities involve `lgamma`, the log of the gamma function, and `lgamma` is where BinaryBackend draws the line.

The Lanczos approximation for `lgamma` is a beautiful piece of numerical analysis: a rational function that achieves fifteen digits of accuracy across the positive reals. Exmc implemented it in pure Nx operations -- nine Lanczos coefficients, a sum of terms of the form `c / (x + i)`, a logarithm, and some arithmetic. Forward evaluation was flawless.

The gradient was not.

Autodiff through the Lanczos approximation requires differentiating `c / (x + i)`, which produces terms of the form `c / (x + i)^2`. When `x + i` is small, these terms are large. When they are very large, Nx BinaryBackend's arithmetic falls through to Elixir's `Complex.divide`, which is not equipped for the kind of numbers that appear in the tails of a Gamma distribution. The result was NaN -- not a number -- which propagated through the gradient, through the leapfrog step, through the tree builder, and into the sampler as a divergent transition.

This was not a bug in the implementation. The Lanczos approximation was correct. The gradient of the Lanczos approximation was mathematically correct. The problem was that BinaryBackend's floating-point arithmetic was not robust enough to evaluate that gradient at extreme values without losing numerical control.

Decision twenty-four was the solution: when EXLA is available, wrap the entire `value_and_grad` computation in `EXLA.jit`. EXLA handles `lgamma` and its gradient natively -- XLA has a dedicated lgamma kernel that never touches the Lanczos coefficients directly. The gradient is computed by the compiler, not by Elixir arithmetic. On EXLA, Gamma and Beta work as sampled priors. On pure BinaryBackend, they work only as observation likelihoods, where their gradient is not needed.

This was the grand compromise of the project: the BinaryBackend path works for everything it can handle, and EXLA swoops in for the operations that require real numerical infrastructure. The detection is automatic. `Code.ensure_loaded?(EXLA)` at compile time. No user intervention needed.

---

## VI. The Fused Step

Decision twenty-seven was born from profiling. Each NUTS step calls the leapfrog integrator, and each leapfrog step calls `value_and_grad`. With EXLA, that call crosses the JIT boundary: tensors go into the XLA runtime, the computation executes on compiled native code, and results come back. The overhead of each crossing is small but nonzero -- a few hundred microseconds for argument marshaling and result extraction.

A single leapfrog step does four things: a half-step momentum update, a full position update, a value-and-gradient evaluation, and another half-step momentum update. If each of these is a separate EXLA call, the overhead multiplies. If the entire leapfrog step is a single EXLA call -- one JIT boundary crossing instead of four -- the overhead is amortized.

`compile_for_sampling` now returns a three-tuple: `{vag_fn, step_fn, pm}`. The `step_fn` fuses the complete leapfrog step into a single XLA computation. Position, momentum, epsilon, gradient, and inverse mass diagonal go in; new position, new momentum, new log-probability, and new gradient come out. One call. The epsilon scalar, which changes during warmup, is wrapped in `Nx.tensor` with BinaryBackend before each call so that EXLA can trace it.

Without EXLA, the sampler falls back to the unfused path: `Leapfrog.step(vag_fn, ...)` with multiple Nx operations. The interface is the same. The speed is not.

---

## VII. The Funnel

The non-centered parameterization was decision thirty-two, and it addressed a problem as old as hierarchical Bayesian modeling itself.

Consider a hierarchical Normal: `mu ~ N(0, 5)`, `sigma ~ HalfNormal(2)`, `x ~ N(mu, sigma)`, with observations pulling `x` toward some value. The posterior over `(mu, sigma, x)` has funnel geometry: when `sigma` is large, `x` can range widely; when `sigma` is small, `x` is tightly constrained. The NUTS sampler, which uses a single global step size, cannot efficiently explore both regimes. Small step sizes waste time in the wide region; large step sizes cause divergences in the narrow region.

The standard solution is non-centered parameterization: instead of sampling `x ~ N(mu, sigma)`, sample `z ~ N(0, 1)` and compute `x = mu + sigma * z`. The funnel disappears because `z` has the same geometry regardless of `sigma`.

In PyMC, the user must apply this transformation manually. In Exmc, it happens automatically. The `NonCenteredParameterization` rewrite pass scans the IR for Normal random variables where both `mu` and `sigma` are string references to other random variables. When it finds one, it rewrites the node: the prior becomes `N(0, 1)`, and the original parameterization is stored in NCP metadata for reconstruction during trace building.

The rewrite is conservative -- it only fires when both parent parameters are free random variables, the situation where funnel geometry is guaranteed. The user never sees it. The trace contains the original parameter `x`, not the auxiliary `z`. But the sampler explores the transformed space where the geometry is benign.

---

## VIII. The Constrained Parent Bug

Of all the bugs in the project, the constrained parent bug was the most insidious because it produced wrong answers that looked right.

The problem was in `resolve_params`, the compiler function that resolves string parameter references. When a distribution says `sigma: "sigma"`, the compiler looks up the value of `sigma` in the current value map. But the value map holds *unconstrained* values. If `sigma` has a `:log` transform -- because it is a HalfNormal or Exponential -- then the value map contains `log(sigma)`, not `sigma`.

For simple models, this bug was invisible. A Normal distribution with `sigma: "sigma"` where sigma's unconstrained value happened to be close to its constrained value (around 1.0, where `log(1) = 0`) would produce a slightly wrong log-probability that was still differentiable and still explored a vaguely reasonable region. The sampler would converge to something, and if you didn't check too carefully, the something looked plausible.

For hierarchical models with five free parameters, the bug was catastrophic. The step-size search would collapse to `1e-141`. Every sample would diverge. The sampler would return a trace of identical values.

The fix was `resolve_params_constrained`: a new function that looks up each string reference's transform in the PointMap and applies the forward transform before returning the value. `log(sigma)` becomes `exp(log(sigma)) = sigma`. The constrained value arrives at the distribution's logpdf as intended.

This was decision twenty-eight, and it partially superseded decision twenty-two. The lesson was that unconstrained space is a useful abstraction for the sampler, but the moment any part of the system needs to think in terms of actual parameter values -- when a child distribution asks what its parent's sigma is -- you must transform back to constrained space. The abstraction has a cost, and the cost is eternal vigilance at the boundary.

---

## IX. Four Chains, One Model

Decision thirty-five was the payoff for everything that came before.

`Sampler.sample_chains(ir, 4, init_values: init)` compiles the model once -- one IR rewrite, one compiler pass, one EXLA JIT compilation -- and dispatches four chains in parallel via `Task.async_stream`. Each chain gets its own `:rand` state, seeded deterministically from the chain index. Each chain produces its own trace and statistics. The four traces come back ordered, ready for convergence diagnostics.

There is no shared mutable state between chains. Erlang's `:rand` uses explicit-state functions -- `uniform_s`, `normal_s` -- that take and return a state value rather than consulting a process dictionary. Nx tensors are immutable by construction. EXLA's JIT closures are thread-safe. The four chains cannot interfere with each other because the language and runtime make interference structurally impossible.

This is the BEAM's thesis, applied to probabilistic programming: concurrency should be so cheap and so safe that you do not think about it. You do not plan for it. You do not debug it. You dispatch four tasks and collect four results and the runtime handles the rest.

On a four-core machine, the wall-clock time for four chains is approximately equal to the time for one chain. Not half, not twice -- approximately equal. Near-linear speedup from a feature that required no locking, no synchronization, no careful reasoning about thread safety. The compile-once architecture means the EXLA JIT compilation -- which can take several seconds for complex models -- happens once and is shared across all chains.

---

## X. Seeing the Chains

The visualization started as a necessity and became a project of its own.

ArviZ is the standard diagnostic toolkit for PyMC. Trace plots, histograms, autocorrelation functions, pair plots, forest plots -- the visual vocabulary of Bayesian inference. In Python, ArviZ renders through Matplotlib, which renders through whatever GUI backend the operating system provides. The plots are static images. They appear in Jupyter notebooks as PNGs.

ExmcViz renders through Scenic, Elixir's native scene graph library. Scenic was designed for embedded systems and information displays -- the kind of always-on, real-time interfaces that run on kiosks and industrial panels. It draws directly through OpenGL via NanoVG. There is no image encoding, no file I/O, no intermediate format. The scene graph is a live data structure that Scenic composites to the framebuffer sixty times per second.

The architecture follows a hard rule: all Nx tensor computation happens in `Data.Prepare`. Components receive plain Elixir lists and floats. This boundary exists because Scenic's rendering pipeline should never wait for tensor operations, and tensor operations should never block the UI thread. `Prepare.from_trace` converts a trace map of Nx tensors into a list of `VarData` structs containing pre-computed samples, histograms, ACF values, ESS estimates, and quantiles. The components -- `TracePlot`, `Histogram`, `AcfPlot`, `SummaryPanel` -- are pure graph builders that arrange primitives according to the data they receive.

The color palette is amber on true black. `{0, 0, 0}` for every background surface. `{255, 176, 0}` for the default line color. Ten chain colors cycling through the warm spectrum: amber, deep orange, gold, burnt orange, tangerine. Red for divergences. Blue for energy transitions. White dots on the forest plot.

True black is not an aesthetic choice. On OLED displays, black pixels are off pixels -- zero power consumption, zero light emission. For a visualization that might run for hours during a long sampling session, the power savings are real. The warm spectrum preserves night-adapted vision, which matters when you are staring at chains at two in the morning trying to understand why the step size collapsed.

![The pair plot renders a k by k grid: histograms on the diagonal, scatter plots below, correlation coefficients above](assets/pair_plot_4k.png)

---

## XI. The Live Wire

The live streaming feature -- `ExmcViz.stream(ir, init, num_samples: 500)` -- connects the sampler directly to the visualization. You call it, a window appears, and you watch the chains fill in sample by sample.

The architecture has three moving parts. A `StreamCoordinator` GenServer sits between the sampler and the Scenic scene. The sampler runs in a background `Task`, calling a special `sample_stream` variant that sends `{:exmc_sample, i, point_map, step_stat}` for each post-warmup draw. The coordinator accumulates these into a growing trace map and a growing stats list. Every ten samples, it flushes the accumulated data to the `LiveDashboard` scene, which rebuilds its entire graph -- recomputing histograms, ACF, summary statistics -- and pushes it to the viewport.

Ten samples per flush is a compromise. One sample per flush would keep the display maximally current but would overwhelm Scenic with graph rebuilds. A hundred samples per flush would be efficient but would make the display feel sluggish. Ten keeps the UI responsive while letting the sampler run at full speed.

The title bar tracks progress: "MCMC Live Sampling (150 / 500)". When the sampler finishes, it becomes "MCMC Live Sampling (complete)". The coordinator forwards the `{:exmc_done, total}` message as `:sampling_complete`, and the scene updates its title one final time.

This is GenServer message passing, the BEAM's native communication primitive. No polling, no shared memory, no callbacks. The sampler sends a message. The coordinator receives it. The coordinator sends a message. The scene receives it. Each process runs on its own schedule, supervised by the runtime, communicating through immutable messages that cannot be corrupted in transit.

![The live dashboard fills in as the sampler runs, amber traces growing from left to right](assets/live_streaming.png)

---

## XII. One Hundred and Twenty-Three Tests

The test suite is the closest thing the project has to a conscience.

One hundred and twenty-three tests: eleven doctests that verify the examples in the documentation actually work, eighty-seven unit tests that cover IR construction, compiler output, transform correctness, rewrite passes, tree building, leapfrog dynamics, mass matrix adaptation, step-size tuning, diagnostics calculations, and predictive sampling, and twenty-five integration tests that run the full sampler on real models.

The integration tests are the ones that matter most and the ones that are hardest to write. MCMC is inherently stochastic. A test that checks whether the posterior mean of a Normal-Normal conjugate model is within 0.1 of the analytical value will fail some percentage of the time, no matter how correct the sampler is. With 500 samples on BinaryBackend, the tolerances have to be generous: mean within 0.3, variance within 1.0 of the target. Tight enough to catch systematic bugs like the variance inflation or the constrained parent error. Loose enough to not fail on random seed 42 when seed 43 would pass.

The integration tests cover the territory: conjugate posteriors where the analytical answer is known, hierarchical models with up to five free parameters, all nine constrained distributions, non-centered parameterization, WAIC and LOO model comparison, vectorized observations, and parallel multi-chain sampling. Each test is a claim that the sampler produces reasonable results for a specific class of model.

Thirty-five architectural decisions recorded in a document. Twelve debugging entries recording bugs found and fixed. One hundred and twenty-three tests asserting that the system works. Thirty-four more tests in ExmcViz covering the data preparation layer.

The project is a prototype. It samples from posteriors. It handles constraints. It adapts its own tuning parameters. It parallelizes across cores. It visualizes its own output in real time on a black screen in amber light.

It is also a bet -- that the BEAM, a virtual machine designed for telephone switches, can learn to do Bayesian statistics, and that the concurrency story that falls out of that runtime is worth the trouble of rebuilding everything else from scratch. Every decision in the document is a record of that bet being placed, one choice at a time, against the reasonable objection that no one asked for a probabilistic programming framework in Elixir.

Nobody asked for the PDP-11 either. Or Erlang. Or the BEAM. The best systems tend to start as answers to questions that nobody was asking, built by people who couldn't stop thinking about the question.

The traces glow amber on black, climbing from left to right, converging on the truth.

---

*Exmc: 123 tests, 35 decisions, 9 distributions, 4 layers, one runtime.*
*ExmcViz: 34 tests, 8 components, 10 chain colors, zero black pixels wasted.*
