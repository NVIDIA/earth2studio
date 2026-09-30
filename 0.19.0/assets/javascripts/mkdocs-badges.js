/** Interactive filters for MkDocs lists, autosummary tables, and API objects. */
(() => {
  "use strict";

  const state = () => window.MKDOCS_BADGES || {
    pages: {}, definitions: {}, style: "rounded", selectable_text: false
  };
  const FILTER_QUERY_PARAMETER = "badge";

  const siteRoot = (() => {
    const scripts = document.querySelectorAll("script[src]");
    for (const script of scripts) {
      const src = script.getAttribute("src") || "";
      if (!src.includes("mkdocs-badges-data.js")) continue;
      try {
        const path = new URL(src, document.baseURI).pathname;
        const marker = "/assets/javascripts/";
        const index = path.indexOf(marker);
        if (index >= 0) return path.slice(0, index + 1);
      } catch (_) { /* Fall through to the configured Material base. */ }
    }
    const config = document.querySelector("#__config");
    try {
      const base = JSON.parse(config?.textContent || "{}").base || ".";
      return new URL(`${base}/`, document.baseURI).pathname;
    } catch (_) { return "/"; }
  })();

  function badgeIdsFromUrl() {
    try {
      const parameters = new URL(window.location.href).searchParams;
      return [...new Set(
        parameters.getAll(FILTER_QUERY_PARAMETER)
          .flatMap((value) => value.split(","))
          .map((value) => value.trim())
          .filter(Boolean)
      )];
    } catch (_) {
      return [];
    }
  }

  function writeBadgeIdsToUrl(active) {
    if (!window.history?.replaceState) return;
    const url = new URL(window.location.href);
    url.searchParams.delete(FILTER_QUERY_PARAMETER);
    active.forEach((badgeId) => {
      url.searchParams.append(FILTER_QUERY_PARAMETER, badgeId);
    });
    window.history.replaceState(window.history.state, "", url);
  }

  function normalisePageName(value) {
    let path = value.split("#", 1)[0].split("?", 1)[0].replaceAll("\\", "/");
    path = path.replace(/^\/+/, "").replace(/^\.\//, "");
    path = path.replace(/\/index\.html$/, "/").replace(/^index\.html$/, "");
    if (path.endsWith(".html")) path = `${path.slice(0, -5)}/`;
    return path;
  }

  function hrefToPageName(href) {
    if (!href || /^(?:#|mailto:|tel:|javascript:)/i.test(href)) return "";
    try {
      let path = decodeURI(new URL(href, document.baseURI).pathname);
      if (path.startsWith(siteRoot)) path = path.slice(siteRoot.length);
      return normalisePageName(path);
    } catch (_) {
      return normalisePageName(href);
    }
  }

  function pageBadges(pageName) {
    const pages = state().pages || {};
    const normalised = normalisePageName(pageName);
    return pages[normalised] || [];
  }

  function makeBadge(badgeId, context = "page") {
    const config = state();
    const definition = (config.definitions || {})[badgeId];
    if (!definition || definition.hidden || (definition.hide_in || []).includes(context)) return null;
    const badge = document.createElement("span");
    badge.className = `mkdocs-badge mkdocs-badge--${config.style || "rounded"}`;
    badge.dataset.badgeId = badgeId;
    badge.dataset.badgeGroup = definition.group || "";
    badge.style.setProperty("--badge-color", definition.color || "#6c757d");
    badge.style.setProperty("--badge-text-color", definition.text_color || "#fff");
    if (definition.tooltip) badge.title = definition.tooltip;
    if (definition.icon) {
      const icon = document.createElement("span");
      icon.className = "mkdocs-badge__icon";
      icon.innerHTML = definition.icon;
      badge.appendChild(icon);
    }
    const displayLabel = context === "filter"
      ? (definition.name || definition.tooltip || definition.label)
      : definition.label;
    if (displayLabel) {
      const label = document.createElement("span");
      label.className = "mkdocs-badge__label";
      label.textContent = displayLabel;
      badge.appendChild(label);
    }
    return badge;
  }

  function directBadgeIds(element) {
    const ids = [];
    element.querySelectorAll(".mkdocs-badge[data-badge-id]").forEach((badge) => {
      const closestObject = badge.closest(".doc-object");
      if (element.classList.contains("doc-object") && closestObject !== element) return;
      if (badge.closest(".mkdocs-badge-filter__controls")) return;
      if (!ids.includes(badge.dataset.badgeId)) ids.push(badge.dataset.badgeId);
    });
    return ids;
  }

  function entryFromLink(element, anchor, tbody = null) {
    const explicit = (element.dataset.badgeIds || "").split(",").filter(Boolean);
    const pageName = element.dataset.pageUrl
      ? normalisePageName(element.dataset.pageUrl)
      : hrefToPageName(anchor?.getAttribute("href") || "");
    return {
      element, anchor, tbody, pageName,
      badgeIds: explicit.length ? explicit : null,
      badgeContext: element.closest("table.mkdocs-badges-autosummary")
        ? "autosummary"
        : "filter",
      renderBadges: element.dataset.renderBadges !== "false"
    };
  }

  function collectEntries(content) {
    const entries = [];
    const managed = new Set();

    content.querySelectorAll("table.mkdocs-badges-autosummary tbody tr").forEach((row) => {
      const anchor = row.querySelector("td:first-child a[href]");
      if (!anchor) return;
      entries.push(entryFromLink(row, anchor, row.parentElement));
      managed.add(row);
    });

    content.querySelectorAll("table:not(.mkdocs-badges-autosummary) tbody tr").forEach((row) => {
      const anchor = row.querySelector("a[href]");
      if (!anchor) return;
      entries.push(entryFromLink(row, anchor, row.parentElement));
      managed.add(row);
    });

    content.querySelectorAll("li").forEach((item) => {
      if (item.parentElement?.closest("li")) return;
      const anchor = item.querySelector("a[href]");
      if (!anchor) return;
      entries.push(entryFromLink(item, anchor));
      managed.add(item);
    });

    content.querySelectorAll(".doc-object").forEach((object) => {
      if (object.classList.contains("doc-class")) return;
      if (object.parentElement?.closest(".doc-object:not(.doc-class)")) return;
      entries.push({
        element: object,
        anchor: null,
        tbody: null,
        pageName: "",
        badgeIds: directBadgeIds(object),
      });
      managed.add(object);
    });

    // A filter can also wrap custom cards or blocks annotated with data-badge-ids.
    content.querySelectorAll("[data-badge-ids]").forEach((element) => {
      if (managed.has(element)) return;
      const anchor = element.querySelector("a[href]");
      entries.push(entryFromLink(element, anchor));
    });
    return entries;
  }

  function badgesForEntry(entry) {
    return entry.badgeIds !== null ? entry.badgeIds : pageBadges(entry.pageName);
  }

  function annotateEntry(entry, badgeOrder) {
    if (!entry.anchor || !entry.renderBadges || directBadgeIds(entry.element).length) return;
    let ids = badgesForEntry(entry).slice();
    if (badgeOrder.length) {
      ids.sort((left, right) => {
        const a = badgeOrder.indexOf(left);
        const b = badgeOrder.indexOf(right);
        return (a < 0 ? badgeOrder.length : a) - (b < 0 ? badgeOrder.length : b);
      });
    }
    const list = document.createElement("span");
    list.className = "mkdocs-badge-list mkdocs-badge-list--entry";
    ids.forEach((id) => {
      const badge = makeBadge(id, entry.badgeContext);
      if (badge) list.appendChild(badge);
    });
    if (!list.children.length) return;
    const cell = entry.anchor.closest("td");
    (cell || entry.anchor.parentElement || entry.element).appendChild(list);
  }

  function isVisible(entry, active, grouped, mode) {
    if (!active.size) return true;
    const ids = badgesForEntry(entry);
    if (!grouped) {
      return mode === "or"
        ? [...active].some((id) => ids.includes(id))
        : [...active].every((id) => ids.includes(id));
    }
    const groups = {};
    active.forEach((id) => {
      const group = id.includes(":") ? id.slice(0, id.indexOf(":")) : "__ungrouped__";
      (groups[group] ||= []).push(id);
    });
    return Object.values(groups).every((members) =>
      members.some((id) => ids.includes(id))
    );
  }

  function applyFilter(entries, active, grouped, mode) {
    entries.filter((entry) => !entry.tbody).forEach((entry) => {
      entry.element.classList.toggle(
        "mkdocs-badge-filter--hidden",
        !isVisible(entry, active, grouped, mode)
      );
    });

    const tableGroups = new Map();
    entries.filter((entry) => entry.tbody).forEach((entry) => {
      if (!tableGroups.has(entry.tbody)) tableGroups.set(entry.tbody, []);
      tableGroups.get(entry.tbody).push(entry);
    });
    tableGroups.forEach((rows, tbody) => {
      rows.forEach((entry) => {
        if (entry.element.parentElement === tbody) tbody.removeChild(entry.element);
      });
      let rowIndex = 0;
      rows.forEach((entry) => {
        if (!isVisible(entry, active, grouped, mode)) return;
        tbody.appendChild(entry.element);
        entry.element.classList.toggle("mkdocs-badge-row--odd", rowIndex % 2 === 0);
        entry.element.classList.toggle("mkdocs-badge-row--even", rowIndex % 2 !== 0);
        rowIndex += 1;
      });
    });
  }

  function setGroupVisibility(widget, group, hidden) {
    const content = widget.querySelector(":scope > .mkdocs-badge-filter__content");
    content?.querySelectorAll(".mkdocs-badge[data-badge-group]").forEach((badge) => {
      if (badge.dataset.badgeGroup === group) {
        badge.classList.toggle("mkdocs-badge--group-hidden", hidden);
      }
    });
    const toggle = [...widget.querySelectorAll(".mkdocs-badge-filter__toggle")]
      .find((button) => button.dataset.badgeGroup === group);
    if (toggle) {
      toggle.setAttribute("aria-pressed", String(hidden));
      toggle.title = `${hidden ? "Show" : "Hide"} ${group} badges`;
    }
  }

  function initialise(widget) {
    if (widget.dataset.badgesReady === "true") return;
    const content = widget.querySelector(":scope > .mkdocs-badge-filter__content");
    if (!content) return;

    const entries = collectEntries(content);
    const grouped = widget.dataset.grouped === "true";
    const mode = widget.dataset.filterMode || "and";
    const badgeOrder = (widget.dataset.badgeOrder || "").split(",").filter(Boolean);
    const labelSource = widget.dataset.filterLabelSource || "auto";
    const useCompactLabels = labelSource === "label" || (
      labelSource === "auto" && Boolean(content.querySelector("table.mkdocs-badges-autosummary"))
    );
    if (useCompactLabels) {
      widget.querySelectorAll(".mkdocs-badge-filter__button").forEach((button) => {
        const definition = (state().definitions || {})[button.dataset.badgeId];
        const label = button.querySelector(".mkdocs-badge__label");
        if (definition && label) label.textContent = definition.label || "";
      });
    }
    const available = new Set(
      [...widget.querySelectorAll(".mkdocs-badge-filter__button")]
        .map((button) => button.dataset.badgeId)
        .filter(Boolean)
    );
    const active = new Set(badgeIdsFromUrl().filter((id) => available.has(id)));
    entries.forEach((entry) => annotateEntry(entry, badgeOrder));

    function sync(updateUrl = false) {
      widget.classList.toggle("mkdocs-badge-filter--active", active.size > 0);
      widget.querySelectorAll(".mkdocs-badge-filter__button").forEach((button) => {
        button.setAttribute("aria-pressed", String(active.has(button.dataset.badgeId)));
      });
      const all = widget.querySelector(".mkdocs-badge-filter__all");
      if (all) all.setAttribute("aria-pressed", String(active.size === 0));
      const clear = widget.querySelector(".mkdocs-badge-filter__clear");
      if (clear) clear.hidden = active.size === 0;
      applyFilter(entries, active, grouped, mode);
      if (updateUrl) writeBadgeIdsToUrl(active);
    }

    widget.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button || !widget.contains(button)) return;
      if (button.classList.contains("mkdocs-badge-filter__button")) {
        const id = button.dataset.badgeId;
        active.has(id) ? active.delete(id) : active.add(id);
        sync(true);
      } else if (
        button.classList.contains("mkdocs-badge-filter__clear") ||
        button.classList.contains("mkdocs-badge-filter__all")
      ) {
        active.clear();
        sync(true);
      } else if (button.classList.contains("mkdocs-badge-filter__toggle")) {
        const hidden = button.getAttribute("aria-pressed") !== "true";
        setGroupVisibility(widget, button.dataset.badgeGroup, hidden);
      }
    });

    (widget.dataset.groupsHidden || "").split(",").filter(Boolean)
      .forEach((group) => setGroupVisibility(widget, group, true));
    widget.dataset.badgesReady = "true";
    sync();
  }

  function initialiseAll() {
    document.documentElement.classList.toggle(
      "mkdocs-badges--no-text-selection",
      state().selectable_text !== true
    );
    document.querySelectorAll(".mkdocs-badge-filter").forEach(initialise);
  }

  if (typeof document$ !== "undefined") document$.subscribe(initialiseAll);
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialiseAll);
  } else {
    initialiseAll();
  }
})();
