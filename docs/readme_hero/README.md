# README hero image

Regenerate with `bash docs/readme_hero/run_all.sh` (needs a GPU, network, and
`EUMETSAT_CONSUMER_KEY` / `EUMETSAT_CONSUMER_SECRET`); output goes to `./hero_out/hero.png`.

The wind globe and its legend share Matplotlib's `cubehelix` palette. Panel headings use
85 px bold type at 88% width, with compact single-line title bands. Both top-row globes
use 1500 × 1500 px boxes; the green NVIDIA wordmark uses 60 px type.
Heading bands are 152 px tall, with subtitles positioned 116 px below the band's top
to leave breathing room beneath the large titles.
