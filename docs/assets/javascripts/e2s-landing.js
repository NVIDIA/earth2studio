(function () {
  const home = document.querySelector("[data-e2s-home]");
  if (!home) return;

  home.classList.add("is-enhanced");

  const canvas = home.querySelector(".e2s-home-canvas");
  const ctx = canvas && canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hero = home.querySelector(".e2s-hero");
  let stars = [];
  let mist = [];
  let animationFrame = 0;
  let lastDrawTime = 0;
  let scrollFrame = 0;
  let scene = { width: 1, height: 1, horizon: 1, heroHeight: 1 };

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function resizeCanvas() {
    if (!canvas || !ctx) return;
    const width = Math.max(1, home.clientWidth);
    const heroHeight = hero ? hero.offsetHeight : window.innerHeight;
    const height = Math.max(window.innerHeight, heroHeight);
    const heroBottom = hero ? hero.offsetTop + hero.offsetHeight : window.innerHeight;
    const horizon = Math.max(1, heroBottom - window.innerHeight * 0.12);
    const dpr = Math.min(window.devicePixelRatio || 1, 1.35);
    scene = { width, height, horizon, heroHeight };
    canvas.style.height = `${height}px`;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const starCount = Math.max(70, Math.min(140, Math.floor((width * height) / 24000)));
    stars = Array.from({ length: starCount }, (_, index) => ({
      x: (index * 179 + 37) % width,
      y: (index * 293 + 61) % height,
      r: 0.45 + ((index * 17) % 120) / 120,
      pulse: (index * 0.73) % Math.PI,
      alpha: 0.28 + ((index * 31) % 70) / 100,
    }));

    const grid = 42;
    const columns = Math.max(1, Math.ceil(width / grid));
    const mistCount = Math.max(32, Math.min(72, Math.floor(width / 20)));
    mist = Array.from({ length: mistCount }, (_, index) => ({
      x: ((index * 5) % columns) * grid + grid * 0.5,
      y: horizon - 20 - ((index * 43) % 300),
      base: horizon,
      speed: 0.12 + ((index * 11) % 26) / 100,
      drift: ((index % 5) - 2) * 0.008,
      r: 1.1 + ((index * 19) % 34) / 10,
      phase: index * 0.41,
    }));
  }


  function indexColor(star, alpha) {
    const greenBias = star.x % 5 < 3;
    return greenBias
      ? `rgba(155, 229, 100, ${alpha * 0.78})`
      : `rgba(206, 235, 255, ${alpha * 0.46})`;
  }

  function drawScene(time = 0) {
    if (!canvas || !ctx) return;
    if (document.hidden) {
      animationFrame = requestAnimationFrame(drawScene);
      return;
    }
    if (lastDrawTime && time - lastDrawTime < 33) {
      animationFrame = requestAnimationFrame(drawScene);
      return;
    }
    lastDrawTime = time;
    const { width, height, horizon } = scene;
    ctx.clearRect(0, 0, width, height);

    for (const star of stars) {
      const twinkle = 0.72 + Math.sin(time * 0.0014 + star.pulse) * 0.28;
      ctx.fillStyle = indexColor(star, star.alpha * twinkle);
      ctx.beginPath();
      ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
      ctx.fill();
    }

    for (let i = 0; i < 6; i += 1) {
      const x = width * (0.08 + i * 0.105) + Math.sin(time * 0.00045 + i) * 28;
      const top = horizon - 410 - Math.sin(i * 1.7) * 60;
      const gradient = ctx.createLinearGradient(x, top, x, horizon + 70);
      gradient.addColorStop(0, "rgba(118, 185, 0, 0)");
      gradient.addColorStop(0.42, i % 2 ? "rgba(118, 185, 0, 0.13)" : "rgba(155, 229, 100, 0.1)");
      gradient.addColorStop(1, "rgba(118, 185, 0, 0)");
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 22 + (i % 3) * 12;
      ctx.beginPath();
      ctx.moveTo(x, horizon + 60);
      ctx.bezierCurveTo(x - 40, horizon - 120, x + 48, horizon - 270, x - 10, top);
      ctx.stroke();
    }

    for (const particle of mist) {
      particle.y -= particle.speed;
      particle.x += particle.drift + Math.sin(time * 0.001 + particle.phase) * 0.035;
      if (particle.y < horizon - 420) {
        particle.y = horizon + 40 + Math.random() * 50;
        particle.x = Math.random() * width;
      }
      if (particle.x < -20) particle.x = width + 20;
      if (particle.x > width + 20) particle.x = -20;
      const progress = clamp((horizon - particle.y) / 420, 0, 1);
      const alpha = (1 - progress) * 0.22;
      ctx.fillStyle = particle.phase % 1.2 < 0.45 ? `rgba(139, 222, 255, ${alpha * 0.52})` : `rgba(155, 229, 100, ${alpha})`;
      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.r * (1 + progress * 2), 0, Math.PI * 2);
      ctx.fill();
    }

    if (!reduceMotion) animationFrame = requestAnimationFrame(drawScene);
  }

  function scaleArtboards() {
    home.querySelectorAll(".e2s-artboard-stage").forEach((stage) => {
      const iframe = stage.querySelector("iframe");
      if (!iframe) return;
      const scale = stage.clientWidth / 1600;
      iframe.style.transform = `scale(${scale})`;
      stage.style.height = `${460 * scale}px`;
    });
  }

  function updateScrollEffects() {
    scrollFrame = 0;
    const heroHeight = scene.heroHeight || window.innerHeight;
    const y = Math.min(window.scrollY, heroHeight);
    const heroProgress = clamp(y / Math.max(heroHeight, 1), 0, 1);
    home.style.setProperty("--e2s-hero-content-y", `${-(y * 0.12)}px`);
    home.style.setProperty("--e2s-hero-panel-y", `${-(y * 0.04)}px`);
    home.style.setProperty("--e2s-planet-y", `${y * 0.1}px`);
    home.style.setProperty("--e2s-hero-opacity", `${clamp(1 - heroProgress * 0.48, 0.72, 1)}`);

  }

  function scheduleScrollEffects() {
    if (!scrollFrame) scrollFrame = requestAnimationFrame(updateScrollEffects);
  }


  function setupInstallSwitcher() {
    const install = home.querySelector("[data-e2s-install]");
    if (!install) return;
    const output = install.querySelector("code");
    const copy = install.querySelector("[data-e2s-copy-command]");
    const buttons = install.querySelectorAll("[data-e2s-command]");
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        buttons.forEach((item) => item.classList.toggle("is-active", item === button));
        const command = button.dataset.e2sCommand || "";
        if (output) output.textContent = command;
        if (copy) copy.dataset.e2sCopyCommand = command;
      });
    });
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

  function setupCopyButtons() {
    home.querySelectorAll("[data-e2s-copy-command]").forEach((button) => {
      const label = button.getAttribute("aria-label") || "Copy command";
      button.addEventListener("click", async () => {
        const command = button.dataset.e2sCopyCommand || "";
        if (!command) return;

        try {
          await writeClipboard(command);
          button.setAttribute("aria-label", "Copied command");
          button.classList.add("is-copied");
        } catch {
          button.setAttribute("aria-label", "Copy failed");
        }

        window.clearTimeout(button.e2sCopyTimer);
        button.e2sCopyTimer = window.setTimeout(() => {
          button.setAttribute("aria-label", label);
          button.classList.remove("is-copied");
        }, 1600);
      });
    });
  }

  function setupPointerGlow() {
    if (reduceMotion) return;
    const glow = document.createElement("span");
    glow.className = "e2s-pointer-glow";
    glow.setAttribute("aria-hidden", "true");
    home.appendChild(glow);

    let frame = 0;
    let x = 0;
    let y = 0;

    function applyPointer() {
      frame = 0;
      glow.style.transform = `translate3d(${x}px, ${y}px, 0) translate3d(-50%, -50%, 0)`;
      glow.style.opacity = "1";
    }

    home.addEventListener("pointermove", (event) => {
      x = event.clientX;
      y = event.clientY;
      if (!frame) frame = requestAnimationFrame(applyPointer);
    });

    home.addEventListener("pointerleave", () => {
      glow.style.opacity = "0";
    });
  }

  window.addEventListener("resize", () => {
    resizeCanvas();
    scaleArtboards();
    scheduleScrollEffects();
  });
  window.addEventListener("scroll", scheduleScrollEffects, { passive: true });

  setupInstallSwitcher();
  setupCopyButtons();
  setupPointerGlow();
  resizeCanvas();
  scaleArtboards();
  updateScrollEffects();
  drawScene();
  window.addEventListener("beforeunload", () => {
    cancelAnimationFrame(animationFrame);
    cancelAnimationFrame(scrollFrame);
  });
})();
