---
hide:
  - navigation
  - toc
---

<!-- markdownlint-disable MD013 MD033 MD036 -->

<div class="e2s-home" data-e2s-home>
  <canvas class="e2s-home-canvas" aria-hidden="true"></canvas>
  <section class="e2s-hero" aria-label="Earth2Studio overview">
    <div class="e2s-planet-scene" aria-hidden="true">
      <div class="e2s-planet"></div>
    </div>
    <div class="e2s-hero__content">
      <p class="e2s-eyebrow">Earth2Studio</p>
      <h1>Experience the next generation of weather and climate modeling</h1>
      <p class="e2s-hero__lede">
        Access a leading collection of weather and climate AI models,
        production-ready data sources,
        composable inference APIs, and GPU-accelerated workflows in one
        Python package.
      </p>
      <a class="e2s-hero__start" href="userguide/about/install/" aria-label="Get started with Earth2Studio">
        <span>Get Started</span>
      </a>
      <div class="e2s-install-command" data-e2s-install aria-label="Install Earth2Studio with FCN support">
        <div class="e2s-install-tabs" role="tablist" aria-label="Package manager">
          <button type="button" class="is-active" data-e2s-command='uv pip install "earth2studio[fcn]"'>UV</button>
          <button type="button" data-e2s-command='pip install "earth2studio[fcn]"'>pip</button>
        </div>
        <div class="e2s-install-line">
          <span>$</span><code>uv pip install &quot;earth2studio[fcn]&quot;</code>
          <button type="button" class="e2s-copy-button" data-e2s-copy-command='uv pip install "earth2studio[fcn]"' aria-label="Copy install command">Copy</button>
        </div>
      </div>
      <div class="e2s-hero__panel" aria-label="Forecast quickstart code example">
        <div class="e2s-terminal">
          <span class="e2s-terminal__dot"></span>
          <span class="e2s-terminal__dot"></span>
          <span class="e2s-terminal__dot"></span>
          <pre><code>from earth2studio import run; from earth2studio.data import GFS
from earth2studio.io import ZarrBackend; from earth2studio.models.px import FCN
model = FCN.load_model(FCN.load_default_package())
run.deterministic([&quot;2024-01-01&quot;], 10, model, GFS(), ZarrBackend(&quot;fcn.zarr&quot;))</code></pre>
        </div>
      </div>
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
      <h2>Forecast with a growing number of models from NVIDIA and the community</h2>
    </div>
    <div class="e2s-showcase-grid">
      <div class="e2s-showcase-card e2s-accent--green e2s-reveal"><strong>FourCastNet 3</strong><span>AFNO-based medium-range forecasting.</span></div>
      <div class="e2s-showcase-card e2s-accent--blue e2s-reveal"><strong>AIFS 2.0</strong><span>ECMWF AI forecast model workflows.</span></div>
      <div class="e2s-showcase-card e2s-accent--blue e2s-reveal"><strong>StormScope</strong><span>Satellite and radar-conditioned forecast workflows.</span></div>
      <div class="e2s-showcase-card e2s-accent--purple e2s-reveal"><strong>HEAL-DA</strong><span>Data assimilation and analysis correction.</span></div>
      <div class="e2s-showcase-card e2s-accent--gold e2s-reveal"><strong>Pangu-Weather</strong><span>Operational-style global forecast rollouts.</span></div>
      <div class="e2s-showcase-card e2s-accent--purple e2s-reveal"><strong>Aurora</strong><span>Foundation-model forecasting and analysis.</span></div>
      <div class="e2s-showcase-card e2s-accent--cyan e2s-reveal"><strong>StormCast-CONUS</strong><span>Regional CONUS forecasting workflows.</span></div>
      <div class="e2s-showcase-card e2s-accent--green e2s-reveal"><strong>CorrDiff</strong><span>Diffusion downscaling workflows.</span></div>
      <div class="e2s-showcase-card e2s-accent--gold e2s-reveal"><strong>DLESyM</strong><span>Coupled Earth-system model inference.</span></div>
      <div class="e2s-showcase-card e2s-accent--cyan e2s-reveal"><strong>GraphCast</strong><span>Global graph neural weather forecasts.</span></div>
      <div class="e2s-showcase-card e2s-accent--cyan e2s-reveal"><strong>ACE-2</strong><span>Allen Institute climate and weather model interface.</span></div>
      <div class="e2s-showcase-card e2s-accent--blue e2s-reveal"><strong>Atlas</strong><span>Generative medium-range forecast workflows.</span></div>
    </div>
    <p class="e2s-more-note e2s-reveal">and more</p>
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
    <p class="e2s-more-note e2s-reveal">and more</p>
  </section>

  <section id="api" class="e2s-band" aria-label="API entry points">
    <p class="e2s-section-kicker">Explore the API</p>
    <h2>Choose the workflow surface you need</h2>
    <div class="e2s-capability-grid">
      <a class="e2s-capability" href="modules/models_px/?badge=class%3Amrf">
        <span>Prognostic</span><strong>Medium range models</strong>
        <small>Global forecast rollouts and ensemble workflows.</small>
      </a>
      <a class="e2s-capability" href="modules/models_dx/?badge=class%3Anwc">
        <span>Prognostic</span><strong>Nowcasting models</strong>
        <small>Rapid-update precipitation, satellite, and radar workflows.</small>
      </a>
      <a class="e2s-capability" href="modules/models_da/?badge=class%3Ada">
        <span>Assimilation</span><strong>Data assimilation models</strong>
        <small>Observation-informed analysis and correction workflows.</small>
      </a>
      <a class="e2s-capability" href="modules/models_px/?badge=class%3Acm&badge=class%3As2s">
        <span>Prognostic</span><strong>Climate models</strong>
        <small>Seasonal, climate, and coupled Earth-system models.</small>
      </a>
      <a class="e2s-capability" href="modules/datasources_analysis/?badge=dataclass%3Aanalysis&badge=dataclass%3Areanalysis">
        <span>Reanalysis</span><strong>Analysis data</strong>
        <small>ERA5, ARCO, HRRR, IFS, and analysis-ready sources.</small>
      </a>
      <a class="e2s-capability" href="modules/datasources_forecast/?badge=dataclass%3Asimulation">
        <span>Numerical</span><strong>Forecast data</strong>
        <small>Operational forecasts for initialization and conditioning.</small>
      </a>
      <a class="e2s-capability" href="modules/datasources_dataframe/?badge=dataclass%3Aobservation">
        <span>DataFrames</span><strong>Observation data</strong>
        <small>Conventional, satellite, radar, and station datasets.</small>
      </a>
      <a class="e2s-capability" href="modules/models_dx/?badge=class%3Ads">
        <span>Diagnostic</span><strong>Downscaling models</strong>
        <small>Super-resolution, correction, and diagnostic downscaling tools.</small>
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
        <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-install</span>
        <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-discover</span>
        <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-data-fetch</span>
        <span><b>$</b> npx skills add NVIDIA/skills --skill earth2studio-deterministic-forecast</span>
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
      <a href="https://www.youtube.com/watch?v=Sog6aCapZeA" target="_blank" rel="noopener">Tutorial</a>
      <a href="userguide/about/install/">Install</a>
      <a href="examples/">Examples</a>
      <a href="modules/">API reference</a>
    </div>
  </section>
</div>

<!-- markdownlint-enable MD013 MD033 MD036 -->
