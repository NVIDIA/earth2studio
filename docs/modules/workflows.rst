:mod:`earth2studio.run`: Workflows
----------------------------------

Built in workflows designed to be a catalyst to help accelerate user defined inference
use cases.

.. warning::
    The built in workflows should not be viewed as silver bullets that work for every
    model and use case. Earth2Studio is focused on enabling users to build and extend,
    these are starting points.

.. automodule:: earth2studio.run
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio

.. autosummary::
    :nosignatures:
    :toctree: generated/workflows/
    :template: function.rst

    run.deterministic
    run.diagnostic
    run.ensemble


:mod:`earth2studio.batched_workflows`: Batched Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for executing deterministic forecast requests with reusable model and
data-source resources.

.. automodule:: earth2studio.batched_workflows
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio.batched_workflows

.. autosummary::
    :nosignatures:
    :toctree: generated/workflows/
    :template: class.rst

    DeterministicBatchRequest
    DeterministicBatchResponse
    DeterministicBatchRuntime

.. autosummary::
    :nosignatures:
    :toctree: generated/workflows/
    :template: function.rst

    run_deterministic_batch
