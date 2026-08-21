(() => {
  const STORAGE_PREFIX = "e2s-announce:";

  // Small stable string hash so dismissals reset automatically when the
  // announcement text changes (mirrors the theme's content-hash behaviour).
  const hash = (value) => {
    let h = 5381;
    for (let i = 0; i < value.length; i += 1) {
      h = ((h << 5) + h + value.charCodeAt(i)) | 0;
    }
    return (h >>> 0).toString(36);
  };

  const storageKey = (el) => {
    const id = el.getAttribute("data-e2s-announce") || "announce";
    const text = (el.textContent || "").replace(/\s+/g, " ").trim();
    return `${STORAGE_PREFIX}${id}:${hash(text)}`;
  };

  const isDismissed = (key) => {
    try {
      return localStorage.getItem(key) === "1";
    } catch (err) {
      return false;
    }
  };

  const remember = (key) => {
    try {
      localStorage.setItem(key, "1");
    } catch (err) {
      /* localStorage unavailable (private mode) — dismiss for this view only */
    }
  };

  // Hide the whole announcement bar once every message inside it is gone, so
  // we never leave an empty coloured banner behind.
  const syncBar = () => {
    const bar = document.querySelector("[data-md-component=announce]");
    if (!bar) return;
    const items = bar.querySelectorAll(".e2s-announcement");
    const anyVisible = Array.prototype.some.call(items, (el) => !el.hidden);
    bar.hidden = items.length > 0 && !anyVisible;
  };

  const init = () => {
    const items = document.querySelectorAll(".e2s-announcement[data-e2s-announce]");
    items.forEach((el) => {
      const key = storageKey(el);
      if (isDismissed(key)) el.hidden = true;

      const button = el.querySelector(".e2s-announcement__close");
      if (button) {
        button.addEventListener("click", () => {
          el.hidden = true;
          remember(key);
          syncBar();
        });
      }
    });
    syncBar();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
