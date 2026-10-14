(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const ease = t => 1 - Math.pow(1 - t, 3);
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];

  // ---- headlines that reveal word by word ----
  $$(".words").forEach(h => {
    const words = h.textContent.trim().split(/\s+/);
    h.innerHTML = words.map((w, i) => `<span class="w" style="transition-delay:${i * 70}ms">${w}</span>`).join(" ");
  });

  // ---- bars ----
  $$(".bars").forEach(c => {
    const max = parseFloat(c.dataset.max);
    c.querySelectorAll(".bar-fill").forEach(f => {
      const pct = Math.max(0.8, parseFloat(f.dataset.v) / max * 100);
      f.dataset.w = pct + "%";
      f.style.backgroundSize = (100 / pct) * 100 + "% 100%";   // the gradient spans the whole track, without repeating
    });
  });

  // ---- interval chart ----
  const num = t => parseFloat(t.replace("−", "-").replace("+", ""));
  $$(".forest").forEach(fp => {
    const min = parseFloat(fp.dataset.min), max = parseFloat(fp.dataset.max), x = v => (v - min) / (max - min) * 100;
    fp.querySelectorAll(".ticks span").forEach(s => { s.style.left = x(num(s.textContent)) + "%"; });
    fp.querySelectorAll(".f-row").forEach(r => {
      const lo = x(+r.dataset.lo), hi = x(+r.dataset.hi), pt = x(+r.dataset.pt), w = Math.max(hi - lo, 0.8);
      const mk = c => Object.assign(document.createElement("div"), { className: c });
      const zero = mk("zero"), ci = mk("ci"), dot = mk("pt");
      zero.style.left = x(0) + "%";
      ci.style.left = lo + "%"; ci.style.width = w + "%"; ci.style.setProperty("--origin", (pt - lo) / w * 100 + "%");
      dot.style.left = pt + "%";
      r.querySelector(".f-axis").append(zero, ci, dot);
    });
  });

  // ---- count-up ----
  const countUp = el => {
    const end = parseFloat(el.dataset.count), dec = +(el.dataset.dec || 0);
    const fmt = v => v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
    if (reduce) { el.textContent = fmt(end); return; }
    const t0 = performance.now();
    const tick = now => { const p = clamp((now - t0) / 1400); el.textContent = fmt(end * ease(p)); if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  };

  // ---- entry reveals (checked against the viewport on scroll) ----
  const show = el => {
    el.classList.add("in");
    el.querySelectorAll(".bar-fill").forEach(f => { f.style.width = f.dataset.w; });
    el.querySelectorAll("[data-count]").forEach(c => { if (!c.dataset.done) { c.dataset.done = 1; countUp(c); } });
    el.querySelectorAll(".f-row").forEach(r => r.classList.add("in"));
  };
  let pending = $$(".reveal, .words, .bars, .f-row");

  // ---- pinned stages ----
  const stageProgress = el => {
    const r = el.getBoundingClientRect(), span = r.height - window.innerHeight;
    return span > 0 ? clamp(-r.top / span) : 1;
  };
  const words = $("[data-stage=words]"), litSpans = $$("#lit span"), capSpans = $$("#litCap span");
  const bar = $("[data-stage=bar]"), meterAfter = $("#meterAfter"), mBefore = $("#mBefore"), mAfter = $("#mAfter"), mBadge = $("#mBadge");
  // top-10 slot strips: 10 squares, filled exactly (partial fill for the last one)
  $$(".slots").forEach(s => { s.innerHTML = "<i></i>".repeat(10); });
  const setRow = (row, target, g) => {   // g: 0..1 growth; only the measured value is ever labelled
    row.querySelector(".mfill").style.width = target * 10 * g + "%";
    [...row.querySelectorAll(".slots i")].forEach((sq, i) => sq.style.setProperty("--f", clamp(target * g - i) * 100 + "%"));
    row.classList.toggle("growing", g > 0 && g < 1);
    row.classList.toggle("done", g >= 1);
  };
  const lap = $("[data-stage=laptop]"), laptop = $("#laptop"), lid = $("#lid");
  const shots = $$(".shot img"), cards = $$(".card-ex"), steps = $$("#lapSteps i");
  const playhead = $("#playhead"), sceneSpan = $("#span"), stLeft = $("#stLeft"), stRight = $("#stRight");

  // the four scenes: start and end second, film length in seconds (1 fps samples), title, status tag
  const EX = [
    { s: 3417, e: 3443, len: 5437, film: "Gone with the West (1975)", tag: "match · literal fit" },
    { s: 2807, e: 2812, len: 3721, film: "Inner Sanctum (1948)", tag: "match · literal fit" },
    { s: 3382, e: 3385, len: 5129, film: "Teenagers from Outer Space (1959)", tag: "match · literal fit" },
    { s: 2081, e: 2085, len: 5437, film: "Gone with the West (1975)", tag: "safety · weapons" },
  ];
  const tc = s => { s = Math.round(s); const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, x = s % 60;
    return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0"); };
  let curEx = -1;
  const setExample = (k, frac) => {
    const ex = EX[k];
    if (k !== curEx) {
      curEx = k;
      shots.forEach((im, i) => im.classList.toggle("on", i === k));
      cards.forEach((c, i) => c.classList.toggle("on", i === k));
      steps.forEach((d, i) => d.classList.toggle("on", i === k));
      sceneSpan.style.left = ex.s / ex.len * 100 + "%";
      sceneSpan.style.width = Math.max(1, (ex.e - ex.s) / ex.len * 100) + "%";
      stRight.textContent = `scene ${k + 1} of 4 · ${ex.tag}`;
    }
    const t = ex.s + (ex.e - ex.s) * frac;
    playhead.style.left = t / ex.len * 100 + "%";
    stLeft.textContent = `${ex.film} · ${tc(t)} / ${tc(ex.len)}`;
  };

  const hero = $("#top"), nav = $("#nav");
  const film = $("#film"), track = $("#track"), frs = $$("#track .fr"), scan = $(".scan");
  const zooms = $$("[data-zoom]");
  const navLinks = $$("#nav a"), navWrap = nav;
  const sections = navLinks.map(a => document.getElementById(a.getAttribute("href").slice(1)));
  const progress = $("#progress");

  const frame = () => {
    ticking = false;
    const y = window.scrollY, vh = window.innerHeight;
    const h = document.documentElement.scrollHeight - vh;
    progress.style.transform = `scaleX(${h > 0 ? y / h : 0})`;

    // reveals
    if (pending.length) {
      pending = pending.filter(el => {
        if (reduce || el.getBoundingClientRect().top < vh * 0.88) { show(el); return false; }
        return true;
      });
    }

    // nav: appears after the hero, highlights the current section
    nav.classList.toggle("show", y > hero.offsetHeight * 0.55);
    let cur = null;
    sections.forEach(s => { if (s.getBoundingClientRect().top <= vh * 0.4) cur = s; });
    navLinks.forEach(a => {
      const on = !!cur && a.getAttribute("href") === "#" + cur.id;
      if (on && !a.classList.contains("active")) navWrap.scrollLeft = a.offsetLeft - (navWrap.clientWidth - a.offsetWidth) / 2;
      a.classList.toggle("active", on);
    });

    if (reduce) return;

    // hero film strip: scrolling speeds it up and tilts it a little more
    filmScroll = y;

    // words light up one by one
    const pw = stageProgress(words);
    const lit = pw < 0.12 ? 0 : pw < 0.38 ? 1 : pw < 0.64 ? 2 : 3;
    litSpans.forEach((s, i) => s.classList.toggle("on", i < lit));
    capSpans.forEach((s, i) => s.classList.toggle("on", i === lit - 1));

    // before bar fills, then the after bar grows past it, then the gain and the ad counts appear
    const pb = stageProgress(bar);
    setRow(mBefore, 4.82, ease(clamp((pb - 0.04) / 0.22)));
    setRow(mAfter, 6.92, ease(clamp((pb - 0.32) / 0.26)));
    mBadge.classList.toggle("on", pb > 0.62);
    meterAfter.classList.toggle("on", pb > 0.76);

    // laptop rises, lid opens, then the screen steps through the examples
    const pl = stageProgress(lap), r = lap.getBoundingClientRect();
    const enter = r.top > 0 ? ease(clamp(1 - r.top / vh)) * 0.6 + ease(clamp(pl / 0.22)) * 0.4 : ease(clamp(0.6 + pl / 0.22 * 0.4));
    laptop.style.transform = `translateY(${(1 - enter) * 140}px) scale(${0.72 + 0.28 * enter})`;
    lid.style.transform = `rotateX(${-72 * (1 - ease(clamp(pl / 0.22)))}deg)`;
    const q = clamp((pl - 0.25) / 0.72) * 4, k = Math.min(3, Math.floor(q));
    setExample(k, clamp(q - k));

    // slow zoom on image tiles
    zooms.forEach(im => {
      const rr = im.parentElement.getBoundingClientRect();
      const t = clamp((vh - rr.top) / (vh + rr.height));
      im.style.setProperty("--z", 1.1 - 0.1 * clamp(t * 1.8));
    });
  };

  // hero film strip drifts on its own; a frame under the red scan line shows what the system decided
  let filmScroll = 0, filmX = 0, lastT = performance.now();
  const loop = track.scrollWidth / 2;           // the frames are listed twice, so the strip can wrap seamlessly
  const drift = now => {
    const dt = Math.min(64, now - lastT); lastT = now;
    if (filmScroll < hero.offsetHeight * 1.2) {
      filmX = (filmX + dt * 0.035) % loop;
      const x = (filmX + filmScroll * 0.6) % loop;
      track.style.transform = `translate3d(${-x}px,0,0)`;
      film.style.setProperty("--tilt", `${(innerWidth > 900 ? -8 : -4) - Math.min(4, filmScroll / 120)}deg`);
      const sx = scan.getBoundingClientRect().left;
      frs.forEach(f => { const r = f.getBoundingClientRect(); f.classList.toggle("hit", r.left - 10 < sx && r.right + 50 > sx); });
    }
    requestAnimationFrame(drift);
  };
  if (reduce) frs.forEach(f => f.classList.add("hit"));
  else requestAnimationFrame(drift);

  let ticking = false;
  const request = () => { if (!ticking) { ticking = true; requestAnimationFrame(frame); } };
  window.addEventListener("scroll", request, { passive: true });
  window.addEventListener("resize", request);

  if (reduce) {   // static: every word lit, final meter value, laptop open on the first example
    litSpans.forEach(s => s.classList.add("on"));
    capSpans.forEach(s => s.classList.add("on"));
    setRow(mBefore, 4.82, 1); setRow(mAfter, 6.92, 1); mBadge.classList.add("on"); meterAfter.classList.add("on");
    setExample(0, 0);
    $$("[data-zoom]").forEach(im => im.style.setProperty("--z", 1));
  } else {
    setExample(0, 0);
  }
  frame();
  setTimeout(frame, 80);
})();
