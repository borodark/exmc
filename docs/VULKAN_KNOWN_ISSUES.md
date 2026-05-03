# Vulkan Backend — Known Issues

Tracked failures observed when running the eXMC test suite under
`EXMC_COMPILER=vulkan` against `Nx.Vulkan.Backend`. Each issue
has a corresponding test (or tests) tagged `:vulkan_known_failure`
in the test files; those tests are auto-excluded by
`test/test_helper.exs` when the Vulkan compiler is selected. Once
an issue is fixed, remove the tag from its test(s) and delete the
corresponding section here.

This document is the *short* reference. The original triage table
that produced the four-mode classification (A: bad ResourceArc,
B: wrong op atom, C: stale spv path, D: type mismatch) lives in
the conversation log archived at
`~/.claude/projects/-home-io-projects-learn-erl-pymc/<session>.jsonl`.

## 1. `reduce_scalar` ArgumentError on cross-module ResourceArc

**Mode**: A (bad ResourceArc)

**Tagged tests**:
- `test/level_set_test.exs` — three Heat2D.solve tests
  (uniform kappa, inclusion-distortion, read_sensors)
- `test/native_tree_test.exs` — two NIF-vs-Elixir equivalence
  tests (simple normal posterior, similar posteriors)

**Symptom**:
```
** (ArgumentError) argument error
  (nx_vulkan 0.0.1) Nx.Vulkan.Native.reduce_scalar(
    #Reference<...>, 0,
    "_build/test/lib/nx_vulkan/priv/shaders/reduce.spv"
  )
  (nx_vulkan 0.0.1) lib/nx_vulkan/backend.ex:346:
    Nx.Vulkan.Backend.do_reduce_f32/5
```

**Root cause** (working hypothesis):
`Nx.Vulkan.Backend.do_reduce_f32/5` calls `to_vulkan!(t)` which
returns a struct whose `.ref` field is supposed to be a
`ResourceArc<VulkanTensor>` — the Rust resource class registered
by nx_vulkan's NIF. In the failing tests, the tensor `t` was
produced by a *different* NIF — the Rust tree builder in
`Exmc.NUTS.Native` (resource class `NUTSTreeState` or similar) or
the `Exmc.LevelSet.Heat2D` solver — and its `.ref` decodes as a
generic `#Reference<>` but does not match
`ResourceArc<VulkanTensor>` when `Nx.Vulkan.Native.reduce_scalar`
tries to receive it. Rustler rejects the term and the
`NifResult<Term<'a>>` returns an `Error::BadArg`, which surfaces
as `(ArgumentError) argument error` with no message.

**Why we believe this**: the `op` argument is `0` (sum, valid),
the spv path resolves to a real shader file (no permission or
existence error), and the same call path works on every other
Vulkan-typed tensor.

**Fix direction** (not yet implemented):
1. Audit `to_vulkan!/1` in `lib/nx_vulkan/backend.ex` to confirm
   what it does when `t.data` is *not* a `Nx.Vulkan.Backend{}`
   struct — it should either round-trip through host memory
   (correct fallback) or raise a clear "this tensor is not
   on the Vulkan backend" error (better).
2. Add an explicit ResourceArc-typecheck NIF (`is_vulkan_tensor?/1`)
   that returns true/false without raising, to use at the
   Elixir-side boundary.
3. Fix the upstream callers — `Heat2D.solve` and the NIF tree
   path — to either keep their tensors on a single backend or
   explicitly transfer through host before re-uploading to Vulkan.

**To reproduce**:
```sh
EXMC_COMPILER=vulkan mix test \
  test/native_tree_test.exs:295 \
  --include vulkan_known_failure
```

## 2. Weibull NUTS sampling — pathologically slow (effectively hangs)

**Tagged test**: `test/weibull_test.exs` — "Weibull RV compiles
and samples via NUTS"

**Symptom**: under `EXMC_COMPILER=vulkan`, the test runs for 1+
hours of CPU time at 97% utilisation without completing the NUTS
warmup phase. ExUnit's `--timeout` flag does not interrupt it
(the BEAM scheduler is occupied on Vulkan NIF dispatches and
ExUnit's timer cannot preempt). Killing the BEAM process is the
only way out.

**Root cause** (working hypothesis):
NUTS warmup runs ~1000 iterations × ~10 leapfrog steps =
~10,000 step_fn calls. Each step_fn call evaluates the model's
log-density and gradient. Weibull's `logpdf` uses `lgamma` (via
the Lanczos series), which expands to a long elementwise chain.
Under Vulkan with the IR walker, each link of the chain becomes
a separate dispatch (~280-700 µs of indirection on top of the
GPU work). Total: 10,000 × ~12 dispatches × ~500 µs ≈ 60
seconds *just for dispatch overhead*, plus the actual compute.

This isn't a bug — it's the predicted consequence of the
break-even rule from `nx_vulkan/RESEARCH_FAST_KERNELS.md`. The
Weibull model at d ≤ 5 is exactly the workload class where
Vulkan loses to EXLA by ~10x; multiplied across thousands of
NUTS iterations, the wall-clock difference is the difference
between "completes in seconds" and "doesn't complete."

**Fix direction (chain-shader pattern proven for Normal)**:
1. ✅ Short term: tag-skip under Vulkan. Done.
2. ✅ The general fix path is verified: a chain shader for the
   distribution that performs K leapfrog steps in one Vulkan
   dispatch reduces per-step overhead from ~6000 µs (12 dispatches
   × ~500 µs) to ~50 µs amortized at K=32. For Normal the chain
   shader (`leapfrog_chain_normal.spv`) ships in `nx_vulkan` and
   posterior variance recovers to within MCMC noise of the EXLA
   reference (`var ≈ 1.03` vs `1.01` on `x ~ N(0,1)`).
3. ⏳ For Weibull specifically: write a fused `leapfrog_chain_weibull.spv`
   following the same template. The Phase 2 set already includes
   chain shaders for Student-t, Cauchy, HalfNormal, Exponential,
   and an f64 Normal sibling. Weibull's `lgamma` expansion makes
   it the heaviest case — exact gradient is closed-form in the
   shape parameter k, so doable. Not yet written.
4. Long term: lower-overhead defn-op dispatch. The ~700 µs per
   `Expr.optional` cost is partly avoidable via callback caching
   per IR node — see "Open research questions" in
   `nx_vulkan/RESEARCH_FAST_KERNELS.md`.

**Status**: this issue is no longer "the structural blocker" —
the chain pattern works. Weibull-specific chain shader is the
remaining work to flip THIS test from skipped to passing under
`EXMC_COMPILER=vulkan`. Tracked as a follow-up.

**To reproduce the original symptom** (warning: takes hours):
```sh
EXMC_COMPILER=vulkan mix test \
  test/weibull_test.exs:98 \
  --include vulkan_known_failure
```

## 3. MvNormal value_and_grad — Vulkan ↔ Defn.Expr backend mixing

**Tagged test**: `test/mv_normal_test.exs` — "MvNormal gradient
via value_and_grad"

**Symptom**:
```
** (RuntimeError) cannot invoke Nx function because it relies on
two incompatible tensor implementations: Nx.Vulkan.Backend and
Nx.Defn.Expr. This may mean you are passing a tensor to defn/jit
as an optional argument or as closure in an anonymous function.
For efficiency, it is preferred to always pass tensors as
required arguments instead. Alternatively, you could call
Nx.backend_copy/1 on the tensor, however this will copy its
value and inline it inside the defn expression.
```

**Root cause**: `Exmc.Dist.MvNormal.logpdf/2` (lib/exmc/dist/mv_normal.ex:24)
captures the precomputed Cholesky / precision matrix in its
closure. Under EXLA the closure is traced normally; under Vulkan
the closure carries a `Nx.Vulkan.Backend` tensor that gets mixed
with the `Nx.Defn.Expr` parameter tensors during gradient tracing.

**Fix direction**: pass the precomputed matrices as explicit
arguments to `logpdf/2` instead of closure captures. Or, lift the
precomputation inside the defn-traced body so it produces a
`Defn.Expr` tensor.

**To reproduce**:
```sh
EXMC_COMPILER=vulkan mix test \
  test/mv_normal_test.exs:99 \
  --include vulkan_known_failure
```

## Related: f32 precision tolerance failures (NOT in this file)

Tests tagged `:requires_f64` (currently `gaussian_random_walk_test.exs`
GRW gradient and `dirichlet_test.exs` Dirichlet gradient) are
*also* skipped under Vulkan, but they're a different class:
correct logic, intolerant tolerance.

These tests assert finite-difference vs autodiff gradient
agreement at `tol = 1e-3` or `tol = 0.01`. Under f64 the
agreement is comfortable; under f32 it exceeds the tolerance
slightly (max diffs of ~5e-3 and ~2e-2 respectively). The fix is
either widen the tolerance conditional on backend precision or
use a different test pattern for f32 backends — but this is a
test-suite tuning question, not a backend bug.

The `:requires_f64` tag is shared with the EMLX backend (also
f32-only) and exists for the same reason.

## When this list goes to zero

When all `:vulkan_known_failure` tags are removed, this file can
be deleted and the corresponding clause in
`test/test_helper.exs` simplified.
