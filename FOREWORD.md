# Translator's Foreword

A translation is an act of admiration that looks, from a distance, like an act of audacity. One does not translate a minor work. One translates the work that would not leave one alone.

PyMC would not leave me alone.

The authors of PyMC — John Salvatier, Thomas V. Wiecki, Christopher Fonnesbeck, and the contributors who have sustained and refined the project across its generations — produced something that transcends its implementation. They established that Bayesian inference could be made ergonomic without being made trivial. That a practitioner could specify a model declaratively, receive correct posterior samples, and diagnose the quality of those samples — all without descending into the manual machinery of Markov chains. This was, and remains, a genuine intellectual achievement. The mathematics were not simplified. The interface was.

What they could not have anticipated — what no one working within the conventions of the Python scientific computing ecosystem had reason to anticipate — is that the architecture they designed contains, within it, an argument for a different runtime. The separation of model specification from sampling execution, the independence of chains, the streaming nature of diagnostics — these are not incidental features of PyMC's design. They are structural properties. And they happen to be the structural properties that the BEAM virtual machine was built to exploit.

eXMC is not a port. A port preserves the surface and discards the argument. eXMC attempts to preserve the argument — that probabilistic programming should be correct, ergonomic, and observable — while discovering what changes when the runtime participates in the architecture rather than merely hosting it. When chains are not threads managed by a pool but lightweight processes supervised by a fault-tolerant tree. When diagnostics are not computed after the fact but streamed as messages between concurrent processes. When a sampling failure does not corrupt shared state but crashes a single process, which its supervisor restarts with known-good parameters.

The BEAM was designed for telephone switches. It was designed for systems where ten thousand concurrent conversations must proceed independently, where any single failure must be isolated, and where the system must continue operating while parts of it are replaced. It was not designed for Hamiltonian Monte Carlo. But the properties that make it reliable for telephony — process isolation, message passing, supervision, hot code replacement — turn out to be precisely the properties that make concurrent Bayesian inference not merely possible but natural.

I owe a debt to the PyMC authors that this foreword can acknowledge but not repay. Every correct distribution in eXMC traces its lineage to their careful implementation. Every diagnostic bears the mark of their thinking about what practitioners need to trust their results. The NUTS sampler in eXMC follows the trajectory they charted; eXMC merely runs it on different ground.

I owe a further debt to José Valim and the creators of Nx and EXLA, who made numerical computation on the BEAM not merely feasible but fast. Without Nx tensors and EXLA's compilation to XLA, eXMC would be a curiosity — correct but unusable. With them, it is an argument.

The question eXMC asks is narrow and specific: what happens when a PyMC-style architecture runs on a fault-tolerant, massively concurrent runtime? The answer, I believe, is interesting. Whether it is also useful is for the practitioner to decide. The translator's foreword is not the place for that verdict. It is the place to say: the original was worth translating, and whatever errors the translation contains are mine alone.

— I.O.
Detroit, 2026
