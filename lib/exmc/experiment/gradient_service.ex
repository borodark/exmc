defmodule Exmc.Experiment.GradientService do
  @moduledoc """
  **EXPERIMENT** — GenServer wrapping a NUTS leapfrog step function.

  Branch: `experiment/dist-default-numerics` (merged 2026-02-12)
  Status: Validated. Zero overhead at >100μs ops. Not yet wired into production sampler.

  ## Results (E24)

  GenServer.call overhead (~5-20μs) is invisible when step_fn takes 200-270μs.
  Full sampling: 1.01x average across simple/medium/stress (within noise).
  Remote dispatch via `:peer` adds ~700μs floor from Nx tensor serialization;
  crossover to <50% overhead at d≈80.

  See `EXPERIMENT_RESULTS.md` for full data, `benchmark/gradient_service_bench.exs`
  to reproduce.

  ## Usage

      {:ok, pid} = GradientService.start_link(step_fn)
      wrapped = GradientService.wrap(pid)
      # wrapped has same arity-5 signature as step_fn
      {q, p, logp, grad, joint} = wrapped.(q, p, grad, eps, inv_mass)

      # Remote dispatch (compile on peer, get back pid):
      {:ok, remote_pid} = :erpc.call(peer_node, GradientService, :start_from_ir, [ir])
      GradientService.step(remote_pid, q, p, grad, eps, inv_mass)

  ## Lessons

  1. `GenServer.start` not `start_link` for `:erpc.call` — ephemeral caller exit kills linked processes
  2. `Nx.backend_copy` to BinaryBackend before replying — EXLA tensors hold node-local handles
  3. No closures across `:erpc` — use `start_from_ir` with serializable IR instead
  """
  use GenServer

  # --- Client API ---

  def start_link(step_fn, opts \\ []) do
    GenServer.start_link(__MODULE__, step_fn, opts)
  end

  @doc "Call step_fn through the GenServer."
  def step(pid, q, p, grad, epsilon, inv_mass) do
    GenServer.call(pid, {:step, q, p, grad, epsilon, inv_mass}, :infinity)
  end

  @doc """
  Return an arity-5 closure that routes through the GenServer.

  Drop-in replacement for step_fn — the tree builder sees a plain function.
  """
  def wrap(pid) do
    fn q, p, grad, epsilon, inv_mass ->
      step(pid, q, p, grad, epsilon, inv_mass)
    end
  end

  @doc """
  Compile a model from IR and start a GradientService wrapping its step_fn.

  Designed for remote dispatch: call via `:erpc.call(node, GradientService, :start_from_ir, [ir])`
  to compile on the remote node and get back only a pid (no closures cross the wire).
  """
  def start_from_ir(ir, compile_opts \\ []) do
    compiled = Exmc.NUTS.Sampler.compile(ir, compile_opts)
    step_fn = elem(compiled, 1)
    # Use start (not start_link) so the process survives the erpc caller exiting
    GenServer.start(__MODULE__, step_fn)
  end

  # --- Server Callbacks ---

  @impl true
  def init(step_fn) do
    {:ok, step_fn}
  end

  @impl true
  def handle_call({:step, q, p, grad, epsilon, inv_mass}, _from, step_fn) do
    {q_new, p_new, logp_new, grad_new, joint_logp} =
      step_fn.(q, p, grad, epsilon, inv_mass)

    # Copy to BinaryBackend so tensors serialize across Erlang distribution.
    # EXLA.Backend tensors hold buffer references only valid on this node.
    result = {
      Nx.backend_copy(q_new, Nx.BinaryBackend),
      Nx.backend_copy(p_new, Nx.BinaryBackend),
      Nx.backend_copy(logp_new, Nx.BinaryBackend),
      Nx.backend_copy(grad_new, Nx.BinaryBackend),
      Nx.backend_copy(joint_logp, Nx.BinaryBackend)
    }

    {:reply, result, step_fn}
  end
end
