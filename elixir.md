# PyMC Architecture Notes for Elixir/Erlang Reimplementation

Below is a focused architecture map based on the local PyMC codebase, with emphasis on matrix computation hotspots and places where BEAM-style concurrency is likely to pay off.

## Architecture Map (Modules and Responsibilities)

1. Model definition + logp/grad compilation
- Model context, RV registration, logp assembly, and PyTensor compilation.
- Main bridge to computation graphs and autodiff.
- Files:
  - `pymc/model/core.py`

2. Tensor/graph utilities
- PyTensor function compilation, graph rewrites, gradients/Hessians, data conversion.
- Files:
  - `pymc/pytensorf.py`

3. Logprob system + graph rewrites
- Logprob derivation from PyTensor graphs and specialized rewrites.
- Matrix-math logprob for matmul with linear solves and slogdet.
- Files:
  - `pymc/logprob/basic.py`
  - `pymc/logprob/rewriting.py`
  - `pymc/logprob/linalg.py`

4. Distributions
- RV constructors, logp definitions, and transforms.
- Heavy linear algebra in multivariate distributions.
- Files:
  - `pymc/distributions/distribution.py`
  - `pymc/distributions/multivariate.py`
  - `pymc/distributions/transforms.py`

5. Sampling orchestration
- MCMC entry point: step method selection, init, chain orchestration, trace output.
- Files:
  - `pymc/sampling/mcmc.py`
  - `pymc/sampling/parallel.py`

6. Step methods
- HMC/NUTS integrators and mass matrices.
- Metropolis proposals and scaling.
- Files:
  - `pymc/step_methods/hmc/quadpotential.py`
  - `pymc/step_methods/hmc/integration.py`
  - `pymc/step_methods/hmc/nuts.py`
  - `pymc/step_methods/metropolis.py`

7. Backends
- Trace storage and ArviZ conversion.
- Files:
  - `pymc/backends/*`

8. Variational inference
- VI/OPVI and minibatch RVs.
- Files:
  - `pymc/variational/*`
  - `pymc/variational/minibatch_rv.py`

## Primary Execution Flow (MCMC)

1. Model definition
- User defines RVs inside `pm.Model()` context. Observed data converted in `pymc/pytensorf.py`.

2. Logp graph construction
- `Model.logp()` collects RV logps and potentials using `transformed_conditional_logp`.

3. Logp + grad compilation
- `Model.logp_dlogp_function()` builds a PyTensor function via `ValueGradFunction`.

4. Step method selection
- `pymc/sampling/mcmc.py` assigns step methods based on variable types and differentiability.

5. Sampling loop
- Each chain calls the step method, which uses the compiled logp/grad function.
- Parallel chains handled in `pymc/sampling/parallel.py` using multiprocessing.

## Matrix Computation Hotspots

1. Logprob graph evaluation (core)
- Matrix ops are embedded in PyTensor graphs, mostly defined in distributions.
- Files:
  - `pymc/distributions/multivariate.py`
  - `pymc/logprob/linalg.py`

2. HMC/NUTS mass matrix and integration
- Cholesky, solves, and dot products in quadpotential and leapfrog integration.
- Files:
  - `pymc/step_methods/hmc/quadpotential.py`
  - `pymc/step_methods/hmc/integration.py`

3. Metropolis proposal scaling
- Cholesky + dot products for correlated proposals.
- File:
  - `pymc/step_methods/metropolis.py`

4. Distribution-level linear algebra
- Cholesky, solves, det, trace, and inverse in multivariate logp.
- File:
  - `pymc/distributions/multivariate.py`

5. Logprob for matmul
- `pymc/logprob/linalg.py` uses linear solves and slogdet for measurable matmul.

## BEAM-Style Concurrency Opportunities

1. Chain-level parallelism (highest ROI)
- PyMC runs chains in separate OS processes. This maps to BEAM process-per-chain.
- Files:
  - `pymc/sampling/parallel.py`
  - `pymc/sampling/mcmc.py`

2. Population methods
- Population stepper supports stepping multiple chains; BEAM can supervise chain workers.
- File:
  - `pymc/sampling/population.py`

3. Independent variable blocks
- Compound/blocked step methods could be parallelized if independence is provable.
- Files:
  - `pymc/step_methods/arraystep.py`
  - `pymc/step_methods/compound.py`

4. Minibatch / VI
- Minibatch and VI updates can benefit from concurrent batch evaluation.
- Files:
  - `pymc/variational/minibatch_rv.py`
  - `pymc/variational/opvi.py`

5. Matrix kernels
- In BEAM, heavy matrix ops should be offloaded to NIFs or a tensor backend (e.g., Nx).
- This is required to avoid scheduler blocking.

## Reimplementation Notes

1. Core graph layer
- PyMC relies on PyTensor for graph IR and autodiff. You need either a similar IR or a bridge to a tensor compiler.

2. Distribution layer
- Start with univariate distributions, then multivariate (`pymc/distributions/multivariate.py`).

3. Logprob engine
- PyMC’s `pymc/logprob/*` is a logp compiler and rewrite system. You need an equivalent in Elixir/Erlang.

4. Sampler layer
- HMC/NUTS: see `pymc/step_methods/hmc/*`.
- Metropolis: see `pymc/step_methods/metropolis.py`.

5. Orchestration
- MCMC loop, progress, trace storage: `pymc/sampling/mcmc.py`.

## Why One VM Beats Separate OS Processes for Chain Parallelism

PyMC uses Python `multiprocessing` to escape the GIL. Each chain runs in a separate OS process, which incurs:

1. **Duplicated model compilation** -- each subprocess pickles/unpickles the compiled PyTensor graph. For complex models this can be seconds per chain.
2. **Duplicated memory** -- every process gets its own copy of the model graph, compiled function, parameter arrays, and Python interpreter state. Memory scales linearly with chain count.
3. **IPC for samples** -- trace results must be serialized through pipes/queues back to the parent. Pure overhead scaling with `n_samples x n_params`.
4. **Process startup cost** -- fork/spawn + interpreter init + module imports + model deserialization. Often dominates short runs.
5. **Coordination is expensive** -- adaptive warmup tuning, convergence diagnostics (R-hat), or early stopping across chains all require cross-process IPC, which is clunky enough that PyMC mostly avoids mid-run coordination.

### What the BEAM gives us

- **Spawn cost**: BEAM process spawn is ~us vs ~ms + interpreter init for OS processes.
- **Shared immutable data**: Compiled closures, point maps, and observed tensors are immutable Elixir terms shared zero-copy across BEAM processes. 4 chains = 1x model memory, not 4x.
- **Parallelism**: Native preemptive scheduling with no GIL. No need for multiprocessing to get true concurrency.
- **Inter-chain messaging**: Lightweight message passing vs serialized IPC through pipes.
- **Coordination**: A supervisor can collect warmup statistics from all chains, compute pooled step size / mass matrix, and broadcast updates. This is the kind of thing NUTS benefits from but PyMC mostly can't do because IPC is too expensive.

### Practical wins

- **Compile once, share everywhere**: `Compiler.compile(ir)` produces `{logp_fn, point_map}`. Spawn N chain processes, hand each the same reference. Zero duplication.
- **Cheap coordination**: Sampler supervisor can pool warmup diagnostics across chains and broadcast tuning updates mid-run.
- **Fast short runs**: For small models or diagnostic reruns, PyMC's per-chain overhead dominates. BEAM process spawn is negligible.
- **Nx backend is orthogonal**: Actual tensor math goes to EXLA/Torchx native code. BEAM handles orchestration (needs concurrency), Nx handles computation (needs speed). Natural split.

### What we don't save

Per-evaluation cost of `logp_fn` is the same -- it's Nx ops either way. The savings are all in the orchestration layer: less memory, less startup, less communication overhead, and the ability to coordinate chains in ways impractical with OS-process isolation.

## Inter-Chain Messaging: What, Why, and What Controls It

There are three categories of inter-chain communication, in order of how much PyMC actually does today.

### 1. Collecting results (PyMC does this)

After sampling, traces from each chain are shipped back to the parent process. This is the minimum -- you need the samples to do anything. In PyMC it goes through `multiprocessing` pipes, serializing numpy arrays.

### 2. Population methods (PyMC does this)

`DEMetropolis` and `DEMetropolisZ` (differential evolution MCMC) make proposals using the current positions of other chains. Chain A proposes its next step based on the difference between Chain B and Chain C's positions. This requires chains to see each other's state every iteration.

In PyMC, `pymc/sampling/population.py` handles this with `PopulationArrayStepShared` using shared memory arrays -- a workaround for the fact that IPC is too slow per-iteration.

### 3. Warmup coordination (PyMC mostly doesn't do this)

This is where the BEAM advantage is largest. NUTS tunes two things during warmup:

- **Step size** (dual averaging) -- each chain adapts independently.
- **Mass matrix** (inverse metric) -- estimated from warmup samples per-chain.

Pooling these across chains would give better estimates faster (4 chains = 4x the warmup samples for mass matrix estimation). Stan and PyMC don't do this because the IPC cost per warmup window isn't worth it. Each chain tunes in isolation and you hope they converge to similar tuning.

Similarly, **online convergence diagnostics** (split-R-hat, ESS) require comparing chain states. Currently computed post-hoc. Running them online would let you stop early when converged or extend when not -- but that requires periodic cross-chain reads.

### What controls it in PyMC

```python
pm.sample(
    draws=1000,
    tune=1000,        # warmup iterations (per-chain, independent)
    chains=4,         # number of chains
    cores=4,          # OS processes for parallelism
    step=pm.NUTS(),   # step method (tunes independently per chain)
)
```

The `parallel.py` module manages the process pool. Each chain is fully independent -- the only message is "here are my final samples" at the end. For population methods, `population.py` uses shared-memory arrays to sidestep IPC.

### What BEAM enables that PyMC can't practically do

- **Pool mass matrix across chains mid-warmup**: Supervisor collects and broadcasts updated matrix. PyMC: too expensive.
- **Online R-hat every N iterations**: Supervisor reads chain states, computes diagnostic. PyMC: not done.
- **Early stopping on convergence**: Supervisor sends `:stop` when R-hat < threshold. PyMC: not done.
- **Adaptive warmup length**: Supervisor extends warmup until pooled diagnostics stabilize. PyMC: fixed `tune=N`.
- **Population MCMC proposals**: Direct message passing, no shared mutable state. PyMC: shared memory workaround.

PyMC's inter-chain messaging is minimal because it's expensive, not because it's undesirable. The BEAM makes it cheap, which opens up coordination patterns that improve sampling efficiency.

