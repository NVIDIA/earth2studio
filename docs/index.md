---
hide:
  - navigation
  - toc
---

<!-- markdownlint-disable MD013 MD033 MD036 -->

<div class="e2s-home" data-e2s-home>
  <canvas class="e2s-home-canvas" aria-hidden="true"></canvas>
  <section class="e2s-hero" aria-label="Earth2Studio overview">
    <div class="e2s-hero__content">
      <div class="e2s-hero__top">
        <div class="e2s-hero__copy">
          <p class="e2s-eyebrow">Earth2Studio</p>
          <h1>Next-generation AI<br>weather modeling</h1>
          <p class="e2s-hero__lede">
            Access a leading collection of weather and climate AI models,
            production-ready data sources,
            composable inference APIs, and GPU-accelerated workflows in one
            Python package.
          </p>
          <a class="e2s-hero__start e2s-home-button" href="userguide/about/install/" aria-label="Get started with Earth2Studio">
            <span>Get Started</span>
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M5 12h14M13 6l6 6-6 6"></path>
            </svg>
          </a>
        </div>
        <div class="e2s-hero__visual">
          <div class="e2s-orbit" data-e2s-orbit role="img" aria-label="Earth2Studio mark: weather and climate icons around a globe">
            <svg class="e2s-orbit__globe" viewBox="0 0 240 240" aria-hidden="true">
              <defs>
                <clipPath id="e2s-orbit-globe-clip">
                  <circle cx="120" cy="120" r="72"></circle>
                </clipPath>
              </defs>
              <circle cx="120" cy="120" r="72"></circle>
              <g clip-path="url(#e2s-orbit-globe-clip)">
                <ellipse cx="120" cy="120" rx="28" ry="72"></ellipse>
                <ellipse cx="120" cy="120" rx="54" ry="72"></ellipse>
                <path d="M48 120h144"></path>
                <path d="M58 88c32 10 92 10 124 0"></path>
                <path d="M58 152c32-10 92-10 124 0"></path>
                <path d="M120 48v144"></path>
              </g>
            </svg>
            <div class="e2s-orbit__icons">
              <button type="button" class="e2s-orbit__icon e2s-home-button e2s-orbit__icon--solar" data-e2s-orbit-label="Solar prediction" aria-label="Solar prediction">
                <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9L17 7M7 17l-2.1 2.1"></path></svg>
              </button>
              <button type="button" class="e2s-orbit__icon e2s-home-button e2s-orbit__icon--cloud" data-e2s-orbit-label="Atmospheric prediction" aria-label="Atmospheric prediction">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 18h11a4 4 0 0 0 .6-7.95A6 6 0 0 0 6.3 8.6 4.5 4.5 0 0 0 6 18z"></path></svg>
              </button>
              <button type="button" class="e2s-orbit__icon e2s-home-button e2s-orbit__icon--storm" data-e2s-orbit-label="Storm nowcasting" aria-label="Storm nowcasting">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2L5 13h6l-1 9 8-11h-6z"></path></svg>
              </button>
              <button type="button" class="e2s-orbit__icon e2s-home-button e2s-orbit__icon--ocean" data-e2s-orbit-label="Ocean modeling" aria-label="Ocean modeling">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 9c2.5-3 4.5-3 7 0s4.5 3 7 0 4.5-3 6 0M2 16c2.5-3 4.5-3 7 0s4.5 3 7 0 4.5-3 6 0"></path></svg>
              </button>
              <button type="button" class="e2s-orbit__icon e2s-home-button e2s-orbit__icon--energy" data-e2s-orbit-label="Renewable energy workflows" aria-label="Renewable energy workflows">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 13v8M9 21h6M12 13V5M12 13l6.9 4M12 13l-6.9 4"></path><circle cx="12" cy="13" r="1.4"></circle></svg>
              </button>
              <button type="button" class="e2s-orbit__icon e2s-home-button e2s-orbit__icon--wind" data-e2s-orbit-label="Wind products" aria-label="Wind products">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 8h9a3 3 0 1 0-3-3M3 12h13a3 3 0 1 1-3 3M3 16h6a2 2 0 1 1-2 2"></path></svg>
              </button>
            </div>
          </div>
        </div>
      </div>
      <div class="e2s-hero__separator" aria-hidden="true"></div>
      <div class="e2s-hero__quickstart">
        <p class="e2s-quickstart-lede">Run an AI forecast with just a few lines of code</p>
        <p class="e2s-github-stats">
          <a href="https://pypi.org/project/earth2studio/" target="_blank" rel="noopener"><img src="https://img.shields.io/pypi/v/earth2studio?style=flat-square&color=76b900&label=release" alt="Latest release"></a>
          <a href="https://github.com/NVIDIA/earth2studio/blob/main/LICENSE" target="_blank" rel="noopener"><img src="https://img.shields.io/pypi/l/earth2studio?style=flat-square&color=76b900" alt="License"></a>
        </p>
        <div class="e2s-install-command e2s-quickstart-card" data-e2s-install aria-label="Install Earth2Studio with FCN support">
          <div class="e2s-install-tabs" role="tablist" aria-label="Package manager">
            <button type="button" class="e2s-home-button is-active" data-e2s-command='uv pip install "earth2studio[fcn]"'>UV</button>
            <button type="button" class="e2s-home-button" data-e2s-command='pip install "earth2studio[fcn]"'>pip</button>
          </div>
          <div class="e2s-install-line">
            <span>$</span><code>uv pip install &quot;earth2studio[fcn]&quot;</code>
            <button type="button" class="e2s-copy-button e2s-home-button" data-e2s-copy-command='uv pip install "earth2studio[fcn]"' aria-label="Copy install command"></button>
          </div>
        </div>
        <div class="e2s-hero__panel" aria-label="Forecast quickstart code example">
          <div class="e2s-terminal e2s-quickstart-card" markdown="1">
            <span class="e2s-terminal__dot"></span>
            <span class="e2s-terminal__dot"></span>
            <span class="e2s-terminal__dot"></span>

```python
from earth2studio import run; from earth2studio.data import GFS
from earth2studio.io import ZarrBackend; from earth2studio.models.px import FCN
model = FCN.load_model(FCN.load_default_package())
run.deterministic(["2024-01-01"], 10, model, GFS(), ZarrBackend("fcn.zarr"))
```

          </div>
        </div>
      </div>
    </div>
  </section>

  <section id="solutions" class="e2s-band e2s-solutions" aria-label="Earth2Studio tools">
    <div class="e2s-section-copy e2s-section-copy--center">
      <p class="e2s-section-kicker">Open platform</p>
      <h2>AI weather and climate tooling for every sector</h2>
      <p>
        Earth2Studio gives research groups, agencies, enterprises, developers,
        and classrooms a shared Python surface for models, data, verification,
        and operational workflows.
      </p>
    </div>
    <div class="e2s-solution-grid">
      <article class="e2s-solution-card">
        <div class="e2s-solution-card__head">
          <span class="e2s-solution-card__icon"><i class="fa-solid fa-flask" aria-hidden="true"></i></span>
          <h3>Scientists & researchers</h3>
        </div>
        <p>
          Benchmark models on identical data through one API, design ensemble
          experiments, and verify outputs with built-in deterministic and
          probabilistic metrics.
        </p>
      </article>
      <article class="e2s-solution-card">
        <div class="e2s-solution-card__head">
          <span class="e2s-solution-card__icon"><i class="fa-solid fa-satellite-dish" aria-hidden="true"></i></span>
          <h3>Met services & agencies</h3>
        </div>
        <p>
          Run, fine-tune, and deploy forecasting capability on infrastructure
          you control, from medium-range global guidance to rapid regional
          workflows.
        </p>
      </article>
      <article class="e2s-solution-card">
        <div class="e2s-solution-card__head">
          <span class="e2s-solution-card__icon"><i class="fa-solid fa-briefcase" aria-hidden="true"></i></span>
          <h3>Enterprise</h3>
        </div>
        <p>
          Build ensemble risk workflows for energy, insurance, logistics,
          agriculture, and climate resilience with reproducible AI forecasts.
        </p>
      </article>
      <article class="e2s-solution-card">
        <div class="e2s-solution-card__head">
          <span class="e2s-solution-card__icon"><i class="fa-solid fa-code" aria-hidden="true"></i></span>
          <h3>Developers</h3>
        </div>
        <p>
          Build weather-aware APIs, dashboards, agents, and decision products
          on a composable SDK that keeps models, data, IO, and workflows
          separate.
        </p>
      </article>
      <article class="e2s-solution-card">
        <div class="e2s-solution-card__head">
          <span class="e2s-solution-card__icon"><i class="fa-solid fa-graduation-cap" aria-hidden="true"></i></span>
          <h3>Educators & students</h3>
        </div>
        <p>
          Teach Earth system AI with an open on-ramp that can fetch data, load
          models, run forecasts, and store outputs using familiar scientific
          Python tools.
        </p>
      </article>
    </div>
  </section>

  <section id="ecosystem" class="e2s-band e2s-ecosystem" aria-label="Integration carousel">
    <div class="e2s-section-copy e2s-section-copy--center">
      <p class="e2s-section-kicker">Open Source Integrated</p>
      <h2>Built on the scientific Python ecosystem</h2>
    </div>
    <div class="e2s-marquee" aria-label="Earth2Studio ecosystem integrations">
      <div class="e2s-marquee__track">
        <div class="e2s-marquee__group">
          <span class="e2s-integration"><b>Za</b>Zarr</span>
          <span class="e2s-integration"><b>Xr</b>Xarray</span>
          <span class="e2s-integration"><b>Cu</b>CuPy</span>
          <span class="e2s-integration"><b>Pa</b>PyArrow</span>
          <span class="e2s-integration"><b>Rp</b>RAPIDS</span>
          <span class="e2s-integration"><b>Pt</b>PyTorch</span>
          <span class="e2s-integration"><b>Ob</b>Obstore</span>
          <span class="e2s-integration"><b>Fs</b>fsspec</span>
        </div>
        <div class="e2s-marquee__group" aria-hidden="true">
          <span class="e2s-integration"><b>Za</b>Zarr</span>
          <span class="e2s-integration"><b>Xr</b>Xarray</span>
          <span class="e2s-integration"><b>Cu</b>CuPy</span>
          <span class="e2s-integration"><b>Pa</b>PyArrow</span>
          <span class="e2s-integration"><b>Rp</b>RAPIDS</span>
          <span class="e2s-integration"><b>Pt</b>PyTorch</span>
          <span class="e2s-integration"><b>Ob</b>Obstore</span>
          <span class="e2s-integration"><b>Fs</b>fsspec</span>
        </div>
      </div>
    </div>
    <p class="e2s-ecosystem-copy">
      AI for weather and climate does not need to feel unfamiliar. If you already
      work with the scientific Python and PyData ecosystem, Earth2Studio gives you
      familiar building blocks for running modern AI weather models.
    </p>
  </section>

  <section class="e2s-band e2s-showcase" aria-label="Model interfaces">
    <div class="e2s-section-copy e2s-section-copy--center">
      <p class="e2s-section-kicker">Model interfaces</p>
      <h2>Forecast with the largest collection of AI models in the community</h2>
    </div>
    <div class="e2s-showcase-grid">
      <div class="e2s-showcase-card e2s-accent--green e2s-reveal"><strong>FourCastNet 3</strong><span>AFNO-based medium-range forecasting</span></div>
      <div class="e2s-showcase-card e2s-accent--blue e2s-reveal"><strong>AIFS 2.0</strong><span>ECMWF AI forecast model workflows</span></div>
      <div class="e2s-showcase-card e2s-accent--blue e2s-reveal"><strong>StormScope</strong><span>Satellite and radar-conditioned forecast workflows</span></div>
      <div class="e2s-showcase-card e2s-accent--purple e2s-reveal"><strong>HEAL-DA</strong><span>Data assimilation and analysis correction</span></div>
      <div class="e2s-showcase-card e2s-accent--gold e2s-reveal"><strong>Pangu-Weather</strong><span>Operational-style global forecast rollouts</span></div>
      <div class="e2s-showcase-card e2s-accent--purple e2s-reveal"><strong>Aurora</strong><span>Foundation-model forecasting and analysis</span></div>
      <div class="e2s-showcase-card e2s-accent--cyan e2s-reveal"><strong>StormCast-CONUS</strong><span>Regional CONUS forecasting workflows</span></div>
      <div class="e2s-showcase-card e2s-accent--green e2s-reveal"><strong>CorrDiff</strong><span>Diffusion downscaling workflows</span></div>
      <div class="e2s-showcase-card e2s-accent--gold e2s-reveal"><strong>DLESyM</strong><span>Coupled Earth-system model inference</span></div>
      <div class="e2s-showcase-card e2s-accent--cyan e2s-reveal"><strong>GraphCast</strong><span>Global graph neural weather forecasts</span></div>
      <div class="e2s-showcase-card e2s-accent--cyan e2s-reveal"><strong>ACE-2</strong><span>AI2 climate and weather model interface</span></div>
      <div class="e2s-showcase-card e2s-accent--blue e2s-reveal"><strong>Atlas</strong><span>Generative medium-range forecast workflows</span></div>
    </div>
    <a class="e2s-more-note e2s-more-note--button e2s-home-button e2s-reveal" href="userguide/about/catalog/?tab=models">and more</a>
  </section>

  <section class="e2s-band e2s-connectors" aria-label="Global data connectors">
    <div class="e2s-section-copy e2s-section-copy--center">
      <p class="e2s-section-kicker">Data connectors</p>
      <h2>Connect to weather and climate data from around the globe</h2>
    </div>
    <div class="e2s-connector-grid">
      <div class="e2s-connector e2s-accent--green e2s-reveal">NOAA</div>
      <div class="e2s-connector e2s-accent--blue e2s-reveal">ECMWF</div>
      <div class="e2s-connector e2s-accent--gold e2s-reveal">NASA</div>
      <div class="e2s-connector e2s-accent--cyan e2s-reveal">EUMETSAT</div>
      <div class="e2s-connector e2s-accent--purple e2s-reveal">EarthMover</div>
      <div class="e2s-connector e2s-accent--green e2s-reveal">Dynamical</div>
      <div class="e2s-connector e2s-accent--blue e2s-reveal">Planetary Computer</div>
      <div class="e2s-connector e2s-accent--cyan e2s-reveal">Copernicus CDS</div>
      <div class="e2s-connector e2s-accent--gold e2s-reveal">NCEP</div>
      <div class="e2s-connector e2s-accent--green e2s-reveal">NCAR</div>
      <div class="e2s-connector e2s-accent--purple e2s-reveal">AWS Open Data</div>
      <div class="e2s-connector e2s-accent--blue e2s-reveal">NNJA</div>
    </div>
    <a class="e2s-more-note e2s-more-note--button e2s-home-button e2s-reveal" href="userguide/about/catalog/?tab=data">and more</a>
  </section>

  <section id="api" class="e2s-band" aria-label="Workflow tools">
    <div class="e2s-section-copy e2s-section-copy--center">
      <p class="e2s-section-kicker">Explore the API</p>
      <h2>Modular components for your use case</h2>
    </div>
    <div class="e2s-capability-grid">
      <a class="e2s-capability e2s-home-button" href="modules/models_px/?badge=task%3Amedium-range">
        <span>Prognostic</span><strong>Medium range models</strong>
        <small>Global forecast rollouts and ensemble workflows</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/models_dx/?badge=task%3Anowcasting">
        <span>Prognostic</span><strong>Nowcasting models</strong>
        <small>Rapid-update precipitation, satellite, and radar workflows</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/models_da/?badge=task%3Adata-assimilation">
        <span>Assimilation</span><strong>Data assimilation models</strong>
        <small>Observation-informed analysis and correction workflows</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/models_px/?badge=task%3Aclimate&badge=task%3Asubseasonal-seasonal">
        <span>Prognostic</span><strong>Climate models</strong>
        <small>Seasonal, climate, and coupled Earth-system models</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/datasources_analysis/?badge=dataclass%3Aanalysis&badge=dataclass%3Areanalysis">
        <span>Reanalysis</span><strong>Analysis data</strong>
        <small>ERA5, ARCO, HRRR, IFS, and analysis-ready sources</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/datasources_forecast/?badge=dataclass%3Asimulation">
        <span>Numerical</span><strong>Forecast data</strong>
        <small>Operational forecasts for initialization and conditioning</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/datasources_dataframe/?badge=dataclass%3Aobservation">
        <span>DataFrames</span><strong>Observation data</strong>
        <small>Conventional, satellite, radar, and station datasets</small>
      </a>
      <a class="e2s-capability e2s-home-button" href="modules/models_dx/?badge=task%3Adownscaling">
        <span>Diagnostic</span><strong>Downscaling models</strong>
        <small>Super-resolution, correction, and diagnostic downscaling tools</small>
      </a>
    </div>
  </section>

  <section class="e2s-band e2s-agent" aria-label="Agent-ready setup">
    <div class="e2s-section-copy e2s-section-copy--center">
      <p class="e2s-section-kicker">Agent ready</p>
      <h2>Automate setup discovery and first forecasts</h2>
      <p>
        Install Earth2Studio skills, then ask your coding agent to recommend a
        model, configure an environment, fetch data, or launch a deterministic
        forecast.
      </p>
    </div>
    <div class="e2s-agent-layout">
      <div class="e2s-agent-terminal e2s-reveal">
        <div class="e2s-agent-terminal__bar" aria-hidden="true">
          <span class="e2s-terminal__dot"></span>
          <span class="e2s-terminal__dot"></span>
          <span class="e2s-terminal__dot"></span>
          <strong>skills.sh</strong>
        </div>
        <div class="e2s-agent-terminal__body">
          <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-install</span>
          <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-discover</span>
          <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-data-fetch</span>
          <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-deterministic-forecast</span>
        </div>
      </div>
      <div class="e2s-agent-grid">
        <div class="e2s-agent-card e2s-accent--cyan e2s-reveal"><strong>Discover</strong><span>Recommend data, models, IO, and docs for a workflow.</span></div>
        <div class="e2s-agent-card e2s-accent--green e2s-reveal"><strong>Install</strong><span>Set up Earth2Studio and model-specific dependencies.</span></div>
        <div class="e2s-agent-card e2s-accent--gold e2s-reveal"><strong>Run</strong><span>Create a forecast with GFS, FourCastNet3, and Zarr output.</span></div>
      </div>
    </div>
  </section>

  <section class="e2s-band e2s-band--split" aria-label="Getting started">
    <div>
      <p class="e2s-section-kicker">Start here</p>
      <h2>Run a forecast, then make it your own</h2>
      <p>
        Earth2Studio keeps the pieces separate: data sources fetch initial states and observations,
        models transform state, IO stores results, and workflows compose them. Easy to get
        started, easy to extend.
      </p>
    </div>
    <div class="e2s-start-grid">
      <a class="e2s-home-button" href="https://www.youtube.com/watch?v=Sog6aCapZeA" target="_blank" rel="noopener">Tutorial</a>
      <a class="e2s-home-button" href="userguide/about/install/">Install</a>
      <a class="e2s-home-button" href="examples/">Examples</a>
      <a class="e2s-home-button" href="modules/">API Reference</a>
    </div>
  </section>
</div>

<!-- markdownlint-enable MD013 MD033 MD036 -->
