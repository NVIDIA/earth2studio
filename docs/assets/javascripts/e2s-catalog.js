(() => {
  const root = document.querySelector("[data-e2s-catalog]");
  if (!root) return;

  const payload = root.querySelector("[data-e2s-catalog-data]");
  const script = document.currentScript || [...document.scripts].find((item) => item.src.endsWith("/e2s-catalog.js"));
  const catalogUrl = script ? new URL("../data/e2s-catalog.json", script.src).toString() : "../../../assets/data/e2s-catalog.json";
  let records = [];
  const params = new URLSearchParams(window.location.search);
  const requestedTab = (params.get("tab") || params.get("catalog") || "model").toLowerCase();
  const state = {
    kind: ["data", "datasource", "datasources", "data-sources"].includes(requestedTab) ? "data" : "model",
    query: params.get("q") || "",
    filters: {},
    page: Math.max(1, Number.parseInt(params.get("page") || "1", 10) || 1),
  };

  const labels = {
    model: "Models",
    data: "Data Sources",
  };
  const groupOrder = {
    model: ["type", "workflow", "source", "framework", "product", "region"],
    data: ["type", "data class", "data family", "source", "product", "region"],
  };
  const pageSize = 10;

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function sortedValues(group) {
    const values = new Set();
    records
      .filter((item) => item.kind === state.kind)
      .forEach((item) => (item.filters[group] || []).forEach((value) => values.add(value)));
    return [...values].sort((a, b) => a.localeCompare(b));
  }

  function selected(group, value) {
    return (state.filters[group] || new Set()).has(value);
  }

  function updateUrl() {
    const next = new URLSearchParams();
    next.set("tab", state.kind === "data" ? "data" : "models");
    if (state.query) next.set("q", state.query);
    if (state.page > 1) next.set("page", String(state.page));
    Object.entries(state.filters).forEach(([group, values]) => {
      [...values].sort().forEach((value) => next.append(group.split(" ").join("_"), value));
    });
    const query = next.toString();
    window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`);
  }

  function hydrateFiltersFromUrl() {
    groupOrder[state.kind].forEach((group) => {
      const values = params.getAll(group.split(" ").join("_"));
      if (values.length) state.filters[group] = new Set(values);
    });
  }

  function matches(item) {
    if (item.kind !== state.kind) return false;
    const haystack = [item.title, item.summary, item.group, ...(item.chips || [])]
      .join(" ")
      .toLowerCase();
    if (state.query && !haystack.includes(state.query.toLowerCase())) return false;

    return Object.entries(state.filters).every(([group, values]) => {
      if (!values.size) return true;
      const itemValues = new Set(item.filters[group] || []);
      return [...values].some((value) => itemValues.has(value));
    });
  }

  function filterPanel() {
    return groupOrder[state.kind]
      .map((group) => {
        const values = sortedValues(group);
        if (!values.length) return "";
        const options = values
          .map((value) => `
            <label class="e2s-catalog-filter${selected(group, value) ? " is-active" : ""}">
              <input type="checkbox" data-filter-group="${escapeHtml(group)}" value="${escapeHtml(value)}" ${selected(group, value) ? "checked" : ""}>
              <span>${escapeHtml(value)}</span>
            </label>`)
          .join("");
        return `
          <fieldset class="e2s-catalog-filter-group">
            <legend>${escapeHtml(group)}</legend>
            ${options}
          </fieldset>`;
      })
      .join("");
  }

  function card(item) {
    const chips = (item.chips || [])
      .map((chip) => `<span>${escapeHtml(chip)}</span>`)
      .join("");
    return `
      <article class="e2s-catalog-card" data-kind="${escapeHtml(item.kind)}" data-tone="${escapeHtml(item.tone)}">
        <div class="e2s-catalog-card__art" aria-hidden="true">
          <span></span>
        </div>
        <div class="e2s-catalog-card__body">
          <div class="e2s-catalog-card__eyebrow">${escapeHtml(item.group)}</div>
          <h3><a href="${escapeHtml(item.url)}">${escapeHtml(item.title)}</a></h3>
          <p>${escapeHtml(item.summary)}</p>
          <div class="e2s-catalog-card__chips">${chips}</div>
        </div>
      </article>`;
  }

  function pagination(pageCount) {
    if (pageCount <= 1) return "";
    return `
      <nav class="e2s-catalog-pagination" aria-label="Catalog pages">
        <button type="button" data-catalog-page="prev" ${state.page === 1 ? "disabled" : ""}>Previous</button>
        <span>Page ${state.page} of ${pageCount}</span>
        <button type="button" data-catalog-page="next" ${state.page === pageCount ? "disabled" : ""}>Next</button>
      </nav>`;
  }

  function render(options = {}) {
    const items = records.filter(matches);
    const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
    state.page = Math.min(Math.max(1, state.page), pageCount);
    const start = (state.page - 1) * pageSize;
    const end = Math.min(start + pageSize, items.length);
    const pageItems = items.slice(start, end);
    const range = items.length ? `Showing ${start + 1}-${end} of ${items.length}` : "0";
    root.innerHTML = `
      <div class="e2s-catalog-toolbar">
        <div class="e2s-catalog-toggle" role="tablist" aria-label="Catalog type">
          <button type="button" data-catalog-kind="model" class="${state.kind === "model" ? "is-active" : ""}">Models</button>
          <button type="button" data-catalog-kind="data" class="${state.kind === "data" ? "is-active" : ""}">Data Sources</button>
        </div>
        <div class="e2s-catalog-search">
          <input type="search" value="${escapeHtml(state.query)}" placeholder="Search ${escapeHtml(labels[state.kind].toLowerCase())}" data-catalog-search>
          <button type="button" data-catalog-clear>Clear filters</button>
        </div>
      </div>
      <div class="e2s-catalog-layout">
        <aside class="e2s-catalog-sidebar">${filterPanel()}</aside>
        <section class="e2s-catalog-results" aria-live="polite">
          <div class="e2s-catalog-count">${range} ${escapeHtml(labels[state.kind].toLowerCase())}</div>
          <div class="e2s-catalog-list">${pageItems.map(card).join("") || '<p class="e2s-catalog-empty">No catalog entries match the selected filters.</p>'}</div>
          ${pagination(pageCount)}
        </section>
      </div>`;
    bind();
    updateUrl();
    if (options.focusSearch) {
      const search = root.querySelector("[data-catalog-search]");
      if (search) {
        search.focus();
        search.setSelectionRange(search.value.length, search.value.length);
      }
    }
    if (options.scrollResults) {
      const results = root.querySelector(".e2s-catalog-results");
      if (results) results.scrollIntoView({ block: "start", behavior: "smooth" });
    }
  }

  function bind() {
    root.querySelectorAll("[data-catalog-kind]").forEach((button) => {
      button.addEventListener("click", () => {
        state.kind = button.dataset.catalogKind;
        state.filters = {};
        state.page = 1;
        render();
      });
    });
    const searchInput = root.querySelector("[data-catalog-search]");
    if (searchInput) searchInput.addEventListener("input", (event) => {
      state.query = event.target.value.trim();
      state.page = 1;
      render({ focusSearch: true });
    });
    const clearButton = root.querySelector("[data-catalog-clear]");
    if (clearButton) clearButton.addEventListener("click", () => {
      state.query = "";
      state.filters = {};
      state.page = 1;
      render();
    });
    root.querySelectorAll("[data-catalog-page]").forEach((button) => {
      button.addEventListener("click", () => {
        state.page += button.dataset.catalogPage === "next" ? 1 : -1;
        render({ scrollResults: true });
      });
    });
    root.querySelectorAll("[data-filter-group]").forEach((input) => {
      input.addEventListener("change", () => {
        const group = input.dataset.filterGroup;
        if (!state.filters[group]) state.filters[group] = new Set();
        if (input.checked) state.filters[group].add(input.value);
        else state.filters[group].delete(input.value);
        if (!state.filters[group].size) delete state.filters[group];
        state.page = 1;
        render();
      });
    });
  }

  async function loadRecords() {
    if (payload && payload.textContent.trim()) {
      records = JSON.parse(payload.textContent || "[]");
      return;
    }
    const response = await fetch(catalogUrl, { credentials: "same-origin" });
    if (!response.ok) throw new Error(`Unable to load catalog data: ${response.status}`);
    records = await response.json();
  }

  loadRecords()
    .then(() => {
      hydrateFiltersFromUrl();
      render();
    })
    .catch(() => {
      root.innerHTML = '<p class="e2s-catalog-empty">Catalog data could not be loaded.</p>';
    });
})();
