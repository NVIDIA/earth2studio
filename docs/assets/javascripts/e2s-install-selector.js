(async function () {
  const script = document.currentScript;
  const root = document.querySelector("[data-e2s-install-selector]");
  if (!root) return;

  const dataElement = root.querySelector("[data-e2s-install-data]");
  let controls = root.querySelector("[data-e2s-install-controls]");
  let output = root.querySelector("[data-e2s-install-output]");
  if (!controls || !output) {
    root.innerHTML = `
      <div class="e2s-install-selector__layout">
        <div class="e2s-install-selector__controls" data-e2s-install-controls></div>
        <div class="e2s-install-selector__output" data-e2s-install-output></div>
      </div>
    `;
    controls = root.querySelector("[data-e2s-install-controls]");
    output = root.querySelector("[data-e2s-install-output]");
  }
  if (!controls || !output) return;

  async function loadData() {
    if (dataElement) {
      return JSON.parse(dataElement.textContent || "{}");
    }

    const fallback = script?.src
      ? new URL("../data/install-options.json", script.src).toString()
      : "assets/data/install-options.json";
    const src = root.dataset.e2sInstallDataSrc || fallback;
    const response = await fetch(src);
    if (!response.ok) {
      throw new Error(`Failed to load install selector data: ${response.status}`);
    }
    return response.json();
  }

  const data = await loadData();
  const params = new URLSearchParams(window.location.search);
  const defaults = data.defaults || {};
  const managers = data.managers || [];
  const sources = data.sources || [];
  const categories = data.categories || [];
  const urlControlsInstall =
    window.location.hash === "#install-command" ||
    ["method", "source", "category", "item"].some((key) => params.has(key));
  let didAutoScroll = false;

  function hasId(items, id) {
    return items.some((item) => item.id === id);
  }

  function categoryById(id) {
    return categories.find((category) => category.id === id) || categories[0];
  }

  function itemById(category, id) {
    return (category.items || []).find((item) => item.id === id) || category.items[0];
  }

  function defaultSource(manager) {
    return defaults.sources?.[manager] || defaults.source;
  }

  const initialManager = hasId(managers, params.get("method"))
    ? params.get("method")
    : defaults.manager;

  const state = {
    manager: initialManager,
    source: hasId(sources, params.get("source")) ? params.get("source") : defaultSource(initialManager),
    category: hasId(categories, params.get("category")) ? params.get("category") : defaults.category,
    item: params.get("item") || defaults.item,
  };

  function escapeHtml(value) {
    return String(value || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function safeHref(value) {
    const href = String(value || "").trim();
    return /^(https?:\/\/|\/|#|\.\.?\/)/.test(href) ? href : "";
  }

  function noteItem(note) {
    if (!note || typeof note !== "object") return escapeHtml(note);

    const text = escapeHtml(note.text || "");
    const href = safeHref(note.href);
    if (!href) return text;

    const label = escapeHtml(note.label || note.href);
    const external = /^https?:\/\//.test(href);
    const target = external ? ' target="_blank" rel="noopener noreferrer"' : "";
    const separator = text ? " " : "";
    return `${text}${separator}<a href="${escapeHtml(href)}"${target}>${label}</a>`;
  }

  function requirement(item) {
    const name = data.package?.name || "earth2studio";
    return item.extra ? `${name}[${item.extra}]` : name;
  }

  function renderTemplate(template, item) {
    return template
      .replaceAll("{requirement}", requirement(item))
      .replaceAll("{github}", data.package?.github || "")
      .replaceAll("{release_ref}", data.package?.release_ref || data.package?.github_ref || "")
      .replaceAll("{main_ref}", data.package?.main_ref || "main")
      .replaceAll("{ref}", data.package?.github_ref || data.package?.release_ref || "main");
  }

  function selectedCommand(item) {
    const template = data.command_templates?.[state.source]?.[state.manager];
    return template ? renderTemplate(template, item) : "";
  }

  function selectedPreinstallCommands(item) {
    return item.preinstall?.[state.manager] || [];
  }

  function writeClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "0";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();

    try {
      const copied = document.execCommand("copy");
      return copied ? Promise.resolve() : Promise.reject(new Error("copy failed"));
    } finally {
      textarea.remove();
    }
  }

  function commandCard(title, commands, description = "", scrollTarget = false) {
    if (!commands.length) return "";
    const commandText = commands.join("\n");
    const scrollAttrs = scrollTarget ? " data-e2s-install-scroll-target" : "";
    return `
      <section class="e2s-install-card"${scrollAttrs}>
        <div class="e2s-install-card__header">
          <div>
            <h3>${escapeHtml(title)}</h3>
            ${description ? `<p>${escapeHtml(description)}</p>` : ""}
          </div>
          <button type="button" data-e2s-install-copy="${escapeHtml(commandText)}">Copy</button>
        </div>
        <pre><code>${escapeHtml(commandText)}</code></pre>
      </section>
    `;
  }

  function autoScrollToInstallCommand() {
    if (didAutoScroll || !urlControlsInstall) return;
    const target = root.querySelector("[data-e2s-install-scroll-target]");
    if (!target) return;
    didAutoScroll = true;
    const scroll = (behavior) => {
      root
        .querySelector("[data-e2s-install-scroll-target]")
        ?.scrollIntoView({ behavior, block: "center" });
    };
    window.requestAnimationFrame(() => scroll("smooth"));
    [120, 300, 700].forEach((delay) => {
      window.setTimeout(() => scroll("auto"), delay);
    });
  }

  function noteList(title, notes, className) {
    if (!notes.length) return "";
    return `
      <div class="${className}">
        <strong>${escapeHtml(title)}</strong>
        <ul>${notes.map((note) => `<li>${noteItem(note)}</li>`).join("")}</ul>
      </div>
    `;
  }

  function buttonGroup(title, name, items, selected) {
    return `
      <fieldset class="e2s-install-step">
        <legend>${escapeHtml(title)}</legend>
        <div class="e2s-install-buttons">
          ${items
            .map(
              (item) => `
                <button
                  type="button"
                  data-e2s-install-field="${escapeHtml(name)}"
                  data-e2s-install-value="${escapeHtml(item.id)}"
                  aria-pressed="${item.id === selected ? "true" : "false"}"
                  class="${item.id === selected ? "is-active" : ""}"
                >
                  <span>${escapeHtml(item.label)}</span>
                  <small>${escapeHtml(item.description || "")}</small>
                </button>
              `
            )
            .join("")}
        </div>
      </fieldset>
    `;
  }

  function itemSelect(category, item) {
    if ((category.items || []).length <= 1) return "";

    return `
      <label class="e2s-install-step e2s-install-select">
        <span>Choose sub-component</span>
        <select data-e2s-install-item>
          ${(category.items || [])
            .map(
              (option) => `
                <option value="${escapeHtml(option.id)}" ${option.id === item.id ? "selected" : ""}>
                  ${escapeHtml(option.label)}
                </option>
              `
            )
            .join("")}
        </select>
      </label>
    `;
  }

  function updateUrl() {
    const next = new URLSearchParams(window.location.search);
    next.set("method", state.manager);
    next.set("source", state.source);
    next.set("category", state.category);
    next.set("item", state.item);
    window.history.replaceState(null, "", `${window.location.pathname}?${next}`);
  }

  function normalizeState() {
    const category = categoryById(state.category);
    state.category = category.id;
    const item = itemById(category, state.item);
    state.item = item.id;
    return { category, item };
  }

  function render() {
    const { category, item } = normalizeState();
    const installCommand = selectedCommand(item);
    const warnings = [
      ...(data.warnings || []),
      ...(category.warnings || []),
      ...(item.warnings || []),
    ];
    const notes = [...(data.notes || []), ...(item.notes || [])];

    controls.innerHTML = `
      ${buttonGroup("1. Choose package manager", "manager", managers, state.manager)}
      ${buttonGroup("2. Choose source", "source", sources, state.source)}
      ${buttonGroup("3. Choose component", "category", categories, state.category)}
      ${itemSelect(category, item)}
    `;

    output.innerHTML = `
      ${noteList("Warnings", warnings, "e2s-install-alert e2s-install-alert--warning")}
      ${noteList("Notes", notes, "e2s-install-alert")}
      ${commandCard("Pre-install steps", selectedPreinstallCommands(item), "Run before the install command when listed.")}
      ${commandCard("Install command", installCommand ? [installCommand] : [], "Copy this command for the selected install target.", true)}
    `;

    autoScrollToInstallCommand();

    root.querySelectorAll("[data-e2s-install-field]").forEach((button) => {
      button.addEventListener("click", () => {
        const field = button.dataset.e2sInstallField;
        const value = button.dataset.e2sInstallValue;
        if (!field || !value) return;
        if (field === "manager") {
          state.manager = value;
          state.source = defaultSource(value);
        }
        if (field === "source") state.source = value;
        if (field === "category") {
          state.category = value;
          state.item = categoryById(value).items[0].id;
        }
        updateUrl();
        render();
      });
    });

    root.querySelector("[data-e2s-install-item]")?.addEventListener("change", (event) => {
      state.item = event.target.value;
      updateUrl();
      render();
    });

    root.querySelectorAll("[data-e2s-install-copy]").forEach((button) => {
      button.addEventListener("click", async () => {
        const text = button.dataset.e2sInstallCopy || "";
        try {
          await writeClipboard(text);
          button.textContent = "Copied";
          button.classList.add("is-copied");
        } catch {
          button.textContent = "Failed";
        }
        window.clearTimeout(button.e2sCopyTimer);
        button.e2sCopyTimer = window.setTimeout(() => {
          button.textContent = "Copy";
          button.classList.remove("is-copied");
        }, 1400);
      });
    });
  }

  render();
})();
