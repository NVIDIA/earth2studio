(function (browser) {
  "use strict";

  const VERSION_SEGMENT = /^(?:main|latest|stable|dev|v|v?\d+(?:\.\d+){1,2}(?:[-._]?[0-9a-z][0-9a-z._-]*)?)$/i;

  function documentationRoot(siteUrl, pageUrl) {
    const configured = new URL(siteUrl, pageUrl.origin);

    const segments = configured.pathname.split("/").filter(Boolean);
    if (segments.length && VERSION_SEGMENT.test(segments[segments.length - 1])) {
      segments.pop();
    }
    const configuredRoot = segments.length ? `/${segments.join("/")}` : "";

    // A local preview can emulate the GitHub project path even though its
    // origin differs. A custom documentation domain instead serves at `/`.
    if (
      pageUrl.pathname === configuredRoot ||
      pageUrl.pathname.startsWith(`${configuredRoot}/`)
    ) {
      return configuredRoot;
    }
    return configured.origin === pageUrl.origin ? configuredRoot : "";
  }

  function mainDocumentationUrl(locationValue, siteUrl) {
    const pageUrl = new URL(locationValue.href);
    return `${pageUrl.origin}${documentationRoot(siteUrl, pageUrl)}/main/`;
  }

  function normalizeMarkdownPath(path) {
    if (/\/(?:index)\.md$/i.test(path)) {
      return path.replace(/index\.md$/i, "");
    }
    if (/\.md$/i.test(path)) return `${path.replace(/\.md$/i, "")}/`;
    return path;
  }

  function redirectTarget(locationValue, siteUrl) {
    const pageUrl = new URL(locationValue.href);
    const root = documentationRoot(siteUrl, pageUrl);
    if (pageUrl.pathname !== root && !pageUrl.pathname.startsWith(`${root}/`)) {
      return null;
    }

    const relativePath = pageUrl.pathname.slice(root.length).replace(/^\/+/, "");
    if (!relativePath) return null;

    const normalizedPath = normalizeMarkdownPath(relativePath);
    const firstSegment = normalizedPath.split("/", 1)[0];
    if (VERSION_SEGMENT.test(firstSegment)) {
      if (normalizedPath === relativePath) return null;
      return `${pageUrl.origin}${root}/${normalizedPath}${pageUrl.search}${pageUrl.hash}`;
    }

    return `${pageUrl.origin}${root}/main/${normalizedPath}${pageUrl.search}${pageUrl.hash}`;
  }

  function currentDocsResourceUrl(value, locationValue, siteUrl) {
    const pageUrl = new URL(locationValue.href);
    const resourceUrl = new URL(value, pageUrl.origin);
    if (resourceUrl.origin !== pageUrl.origin) return value;

    const root = documentationRoot(siteUrl, pageUrl);
    for (const directory of ["assets", "_static"]) {
      const source = `${root}/${directory}/`;
      if (resourceUrl.pathname.startsWith(source)) {
        resourceUrl.pathname = `${root}/main/${directory}/${resourceUrl.pathname.slice(source.length)}`;
        return resourceUrl.href;
      }
    }
    return value;
  }

  function rebaseCurrentDocsResources(documentValue, locationValue, siteUrl) {
    const rewrite = function (element) {
      for (const attribute of ["href", "src"]) {
        if (!element.hasAttribute || !element.hasAttribute(attribute)) continue;
        const value = element.getAttribute(attribute);
        const rebased = currentDocsResourceUrl(value, locationValue, siteUrl);
        if (rebased !== value) element.setAttribute(attribute, rebased);
      }
    };

    documentValue.querySelectorAll("link[href], script[src], img[src]").forEach(rewrite);
    new MutationObserver(function (records) {
      for (const record of records) {
        for (const node of record.addedNodes) {
          if (node.nodeType !== 1) continue;
          rewrite(node);
          node.querySelectorAll?.("link[href], script[src], img[src]").forEach(rewrite);
        }
      }
    }).observe(documentValue.documentElement, { childList: true, subtree: true });
  }

  const api = {
    currentDocsResourceUrl,
    documentationRoot,
    mainDocumentationUrl,
    normalizeMarkdownPath,
    rebaseCurrentDocsResources,
    redirectTarget,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (!browser || !browser.location || !browser.document) return;

  const script = browser.document.currentScript;
  const siteUrl = script && script.dataset.siteUrl ? script.dataset.siteUrl : "/";
  rebaseCurrentDocsResources(browser.document, browser.location, siteUrl);
  const target = redirectTarget(browser.location, siteUrl);
  if (target && target !== browser.location.href) {
    browser.location.replace(target);
    return;
  }

  browser.document.addEventListener("DOMContentLoaded", function () {
    const mainUrl = mainDocumentationUrl(browser.location, siteUrl);
    browser.document.querySelectorAll("[data-e2s-main-docs]").forEach(function (link) {
      link.href = mainUrl;
    });
  });
})(typeof window !== "undefined" ? window : undefined);
