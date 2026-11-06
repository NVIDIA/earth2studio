<!-- markdownlint-disable MD033 -->

# Catalog

Use **Available GPU VRAM** on the Models tab to show models whose known recommended
GPU memory is at or below your available capacity. For example, **80 GB** includes
recommendations of 24, 40, 48, and 80 GB, but excludes 96 GB. The choices come from
model metadata; each model card keeps its recommended VRAM visible. This filter
combines with search and the other filters, and is saved in the page URL as `vram`.
Choose **Any capacity** to remove only the VRAM cutoff, or **Clear filters** to reset
all filters and search. Data Sources are unaffected.

Models without an unambiguous numeric GB recommendation are labeled
**Recommended VRAM: unknown** and excluded whenever a capacity is selected. This
includes missing or malformed values, ranges, multiple recommendations, CPU-only
labels, and multi-GPU configurations. These entries remain visible with **Any
capacity**. An explicit numeric zero is included at any capacity; a CPU label is
not assumed to mean zero. Capacities are neither inferred from hardware names nor
summed across GPUs. Unsupported or invalid `vram` URL values reset to Any capacity.

Recommendations do not guarantee hardware compatibility or performance. Consult
each model's documentation for its requirements.

<section class="e2s-catalog" data-e2s-catalog>
  <p class="e2s-catalog__loading">Loading catalog...</p>
</section>
