.. _nvcoupler_examples:

Coupling (nvcoupler)
--------------------

Examples for nvcoupler, the NUOPC/ESMF-inspired coupling framework for AI
Earth-system inference. All examples run on synthetic toy components — no
model weights, GPU, or network access required — and each prints
hand-verifiable numbers. They cover the coupled atmos/ocean loop declared as
a coupling graph (with a windowed-reduction connector), coupling order
experiments via explicit run sequences, impact chains mixing windowed
connectors and accumulation mediators, vertical (hybrid to pressure)
coupling for chemistry, gradient flow across the exchange for coupled
fine-tuning, and pull-pattern (StormCast-style) conditioning via
``PullAdapter``.
