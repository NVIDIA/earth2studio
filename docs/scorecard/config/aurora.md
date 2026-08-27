---
label: Aurora
category: Prognostic models
px_class: Aurora
short: Aurora is a foundation model of the atmosphere from Microsoft Research.
---

Aurora is a foundation model of the atmosphere from Microsoft Research: a
1.3B-parameter Swin-transformer with Perceiver-style encoders pretrained on
over a million hours of diverse weather and climate data. The version scored
here is the 0.25° deterministic medium-range configuration, which consumes the
two most recent analysis frames (t-6h and t0) and steps forward 6 hours at a
time on a 720x1440 grid (pole-padded onto ERA5's 721x1440 for verification).

## Reference

Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Brandstetter, J.,
Garvan, P., ... & Perdikaris, P. (2024). Aurora: A foundation model of the
atmosphere. arXiv preprint arXiv:2405.13063, 1(8).
