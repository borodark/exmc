# Decisions

This document records key architectural and design decisions for the Exmc prototype.

## 1. Nx as the numeric backend
- Decision: Use Nx for tensor operations, broadcasting, and autodiff primitives.
- Rationale: Nx provides a pure Elixir API with backends for CPU/GPU, and integrates with Defn/EXLA later.
- Implication: All arithmetic in the codebase must use `Nx.*` operators, not `Kernel` arithmetic.

## 2. Minimal probabilistic IR
- Decision: Represent models as a small IR (`Exmc.IR` + `Exmc.Node`) with RV, obs, and det nodes.
- Rationale: Keeps logprob construction explicit and composable before introducing a full graph compiler.
- Implication: No implicit inference; logprob is derived from explicit nodes.

## 3. Rewrite pipeline with named passes
- Decision: Implement a rewrite pipeline with named passes and a `Pass` behavior.
- Rationale: Mirrors PyMC/PyTensor rewrite systems, keeps transforms/measurable ops modular.
- Implication: All structural changes to the IR should be modeled as passes.

## 4. Default transforms by distribution metadata
- Decision: Distributions declare their default transform (`:log`, `:softplus`, `:logit`).
- Rationale: Centralizes constraints per distribution and mirrors PyMC transforms.
- Implication: Transform handling is automatic unless overridden.

## 5. Observations carry metadata
- Decision: Observations carry metadata (likelihood, weight, mask, reduce).
- Rationale: Enables weighted likelihoods, masking, and aggregation without extra graph nodes.
- Implication: Logprob application must honor metadata.

## 6. Measurable ops lifted by rewrite
- Decision: Measurable `matmul` and `affine` are rewritten into `:meas_obs` nodes.
- Rationale: Keeps measurable logic out of core obs handling and mirrors PyMC logprob rewrites.
- Implication: Future measurable ops should be implemented as passes.

## 7. Deterministic nodes do not contribute to logp
- Decision: Deterministic nodes only feed obs or other dets, they add no logprob directly.
- Rationale: Matches probabilistic semantics and PyMC behavior.

## 8. Tests prioritize numeric correctness
- Decision: All tests use explicit Nx expressions and compare numerically with tolerance.
- Rationale: Avoids Kernel arithmetic and ensures numeric parity for logprob terms.

## 9. Free RVs identified by exclusion
- Decision: Free RVs are RV nodes not targeted by any obs/meas_obs node.
- Rationale: Simple, robust identification — no extra metadata needed on nodes.

## 10. Flat vector holds unconstrained values
- Decision: The sampler's flat vector stores unconstrained values; transforms are applied inside the logp function.
- Rationale: Samplers (HMC, NUTS) operate in unconstrained space. Transforms inside logp keep the interface clean.

## 11. Compiler pre-dispatches at build time
- Decision: The compiler walks IR nodes once at build time, producing closures that are pure Nx ops at runtime.
- Rationale: Elixir-level dispatch (pattern matching, map lookups) happens once; the returned logp_fn is a chain of Nx ops traceable by Nx.Defn.grad.

## 12. Obs terms computed eagerly
- Decision: Observation logprob terms are computed eagerly at compile time as constant tensors.
- Rationale: Observed values are constant w.r.t. free RVs in the current IR, so their logprob contribution is fixed.

## 13. Free RVs sorted alphabetically for deterministic layout
- Decision: Free RVs are sorted alphabetically by id in the PointMap.
- Rationale: Ensures deterministic flat-vector layout across runs, independent of map insertion order.

## 14. NUTS uses plain Elixir + Nx, no defn
- Decision: The NUTS sampler is implemented with plain Elixir functions and Nx tensor ops, not `defn`.
- Rationale: Only BinaryBackend is available; `defn` adds tracing complexity without JIT benefit.

## 15. Sampler operates in unconstrained space, returns constrained trace
- Decision: The sampler operates entirely in unconstrained (flat f64) space and applies forward transforms when building the trace.
- Rationale: Consistent with D10. HMC/NUTS require unconstrained geometry; users expect constrained outputs.

## 16. PRNG via Erlang `:rand` with deterministic seeding
- Decision: Sampler uses `:rand.seed_s(:exsss, seed)` for all random decisions (direction, proposal, momentum).
- Rationale: `Nx.Random.split/uniform` are prohibitively slow with BinaryBackend due to defn tracing overhead. `:rand` is fast, deterministic, and reproducible given the same seed.

## 17. Scalar math for adaptation, Nx for geometry
- Decision: Dual averaging and Welford use Erlang `float` arithmetic; leapfrog/KE/momentum use `Nx.t()`.
- Rationale: DA is simple scalar arithmetic that doesn't benefit from tensors. Leapfrog must compose with autodiff.

## 18. Multinomial NUTS (not slice-based)
- Decision: Tree building uses multinomial sampling for proposal selection, not the original slice-based method.
- Rationale: Modern standard per Betancourt 2017. Better exploration and simpler implementation.

## 19. Diagonal mass matrix only
- Decision: Mass matrix adaptation uses diagonal (element-wise variance) only.
- Rationale: Sufficient for prototype; dense mass matrix can be added later without API changes.

## 20. Stan-style three-phase warmup with doubling windows
- Decision: Warmup uses three phases (step size only, step size + mass matrix with doubling windows, step size only).
- Rationale: Proven effective schedule from Stan. Doubling windows allow the mass matrix to stabilize progressively.

