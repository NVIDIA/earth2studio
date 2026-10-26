.. _nvcoupler_examples:

Coupling (nvcoupler)
--------------------

Examples for ``earth2studio.nvcoupler``, which advances ordinary prognostic and
diagnostic models on one clock and serves each model's declared forcing from
the others' outputs. All examples run on toy models from
``earth2studio.nvcoupler.testing`` (no weights, GPU or network access) and print
hand-verifiable numbers. They cover the coupled atmosphere/ocean loop with a
windowed-mean exchange, lagged versus same-step coupling as a matter of what the
destination declares, an impact chain of diagnostics over time windows, running
a coupled system under ``Pipeline``, and hybrid-to-pressure vertical
interpolation in a connector transform.
