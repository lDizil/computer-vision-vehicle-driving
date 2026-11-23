"""Inline layout and browser-only player for the local camera comparison."""

DOCUMENT_START = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Видеосравнение · OpenScene</title>
<style>
:root{color-scheme:light;--ink:#142c3d;--muted:#4f6978;--teal:#135d59;--edge:#d7e5ed;--paper:#f2f7fa}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font:16px/1.48 system-ui,-apple-system,sans-serif}main{max-width:1060px;margin:auto;padding:30px 22px 48px}
.hero{background:linear-gradient(123deg,#142c45,#1b555c);color:#f6fcff;border-radius:20px;padding:26px 30px;
box-shadow:0 12px 26px #1b485d22}.eyebrow{color:#afdfdc;font-size:.76rem;letter-spacing:.11em;font-weight:750;margin:0}
h1,h2,p{overflow-wrap:anywhere}h1{font-size:clamp(1.75rem,4.8vw,3rem);line-height:1.08;margin:.3em 0 .42em}
.hero p{max-width:74ch}.chips{display:flex;flex-wrap:wrap;gap:8px}.chips span{font-size:.84rem;background:#ffffff18;
border:1px solid #ffffff29;padding:6px 9px;border-radius:99px}
.controls,.panel,.legend{background:#fff;border:1px solid var(--edge);border-radius:14px;padding:14px;margin-top:16px}
.selectors{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr) minmax(88px,.27fr);gap:12px}
label{display:block;font-weight:650;font-size:.88rem;color:#244557}
button,select,input{font:inherit;accent-color:var(--teal)}select{display:block;width:100%;max-width:100%;padding:7px 8px;
border:1px solid #b8ced9;border-radius:8px;background:#f8fcfe;color:var(--ink);margin-top:4px}
.playback{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px}
button{padding:6px 11px;border-radius:8px;border:1px solid #b8ced9;color:#143d49;background:#f7fbfd;cursor:pointer}
button[disabled]{cursor:default;opacity:.6}button.primary{background:var(--teal);color:white;border-color:var(--teal);min-width:116px}
:focus-visible{outline:3px solid #218e87;outline-offset:3px}.scrub{flex:1;min-width:155px;margin:0 2px 0 6px}
input[type=range]{display:block;width:100%;margin:3px 0}.readout{margin:.58em 0 0;font-feature-settings:"tnum"}
.hint,summary{font-size:.88rem;color:var(--muted)}kbd{border:1px solid #d4e1e7;border-radius:4px;padding:1px 4px}
.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.panel h2{font-size:1.02rem;margin:0 0 9px}.stage{position:relative;aspect-ratio:1;background:#1d3340;
border-radius:9px;overflow:hidden}.stage canvas,.stage svg{position:absolute;inset:0;width:100%;height:100%;display:block}
.stage svg{pointer-events:none}.pred-box,.truth-box,.guide{fill:none;vector-effect:non-scaling-stroke}
.pred-box{stroke:#73ffe1;stroke-width:2.4}.truth-box{stroke:#ffd268;stroke-width:2.2;stroke-dasharray:5 3}
.guide{stroke:#eef9ff;stroke-width:1;opacity:.34;stroke-dasharray:2 4}.shade{fill:#ccd7dd;fill-opacity:.25}
.zones{list-style:none;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;padding:0;margin:10px 0 0}
.zone{padding:6px 7px;border-radius:8px;border:1px solid #d9e8ee;background:#f6fbfd;min-width:0}
.zone[data-state="1"]{background:#e4f6ef;border-color:#c5eadb}.zone[data-state="0"]{background:#eef2f4;border-color:#dce4e9}
.zone b,.zone span{display:block;overflow-wrap:anywhere}.zone b{font-size:.83rem}.zone span{color:#385563;font-size:.77rem}
.reset{min-height:1.35em;margin:7px 0 0}.legend{display:flex;gap:11px 20px;flex-wrap:wrap}.legend h2{margin:0;font-size:.92rem}
.legend p{margin:0}.legend i{display:inline-block;width:24px;margin-right:6px;vertical-align:middle}
.gold{border-top:3px dashed #ad6d11}.mint{border-top:3px solid #15775e}.gray{height:9px;background:#aab8c1;border-radius:2px}
.caveats{margin:15px 3px 0;color:var(--muted);font-size:.9rem}.caveats p{margin:.45em 0}
.error{margin-top:14px;color:#8e3128}
@media (max-width:730px){main{padding:18px 13px 34px}.hero{padding:19px}.selectors{grid-template-columns:1fr 1fr}}
@media (max-width:540px){.panels,.selectors{grid-template-columns:1fr}.panel{margin-top:11px}.scrub{flex-basis:100%;margin:0}
.legend{gap:8px 12px}.hero{border-radius:14px}}
</style>
</head>
<body>
<main>
<header class="hero">
<p class="eyebrow">MV / OPENSCENE / CAM_F0</p>
<h1>Как меняются решения между кадрами</h1>
<p>Слева — покадровое решение, справа — другой режим на том же кадре. Рамки и «не уверен» можно сравнивать в движении или покадрово.</p>
<div class="chips"><span id="totals">Подготовленные клипы</span><span>Один и тот же снимок слева и справа</span>
<span>Без сети · офлайн-анализ</span></div>
</header>
<section class="controls" aria-label="Управление сравнением">
<div class="selectors">
<label>Клип<select id="clip"></select></label>
<label>Режим справа<select id="mode"></select></label>
<label>Скорость просмотра<select id="rate"><option value="0.5">0,5×</option><option value="1" selected>1×</option>
<option value="2">2×</option></select></label>
</div>
<div class="playback">
<button type="button" id="prev" aria-label="Предыдущий кадр" title="Предыдущий кадр">←</button>
<button type="button" class="primary" id="play" aria-pressed="false">Воспроизвести</button>
<button type="button" id="next" aria-label="Следующий кадр" title="Следующий кадр">→</button>
<label class="scrub">Кадр<input type="range" id="frame" min="0" max="0" value="0" step="1"></label>
</div>
<p id="readout" class="readout"></p>
<p id="dev" class="hint"></p>
<p id="error" class="error" role="alert" hidden></p>
</section>
<div class="panels">
<section class="panel" aria-labelledby="left-title"><h2 id="left-title">Покадровый</h2>
<div class="stage"><canvas id="left-image" width="224" height="224" role="img" aria-label="Подготовленный кадр"></canvas>
<svg id="left-overlay" viewBox="0 0 224 224" aria-hidden="true"></svg></div>
<ul id="left-zones" class="zones" aria-label="Решения слева"></ul>
<p id="left-reset" class="hint reset"></p></section>
<section class="panel" aria-labelledby="right-title"><h2 id="right-title">Сравниваемый режим</h2>
<div class="stage"><canvas id="right-image" width="224" height="224" role="img" aria-label="Тот же подготовленный кадр"></canvas>
<svg id="right-overlay" viewBox="0 0 224 224" aria-hidden="true"></svg></div>
<ul id="right-zones" class="zones" aria-label="Решения справа"></ul>
<p id="right-reset" class="hint reset"></p></section>
</div>
<section class="legend" aria-label="Обозначения"><h2>На кадре</h2><p><i class="gold"></i>Проекция аннотации</p>
<p><i class="mint"></i>Рамка, если модель ответила «есть»</p><p><i class="gray"></i>«Не уверен»</p></section>
<div class="caveats"><p><kbd>←</kbd> <kbd>→</kbd> — смена кадра, <kbd>пробел</kbd> — пуск и пауза.
Пауза между кадрами следует времени исходной записи, а не времени вычисления модели.</p>
<p>Эталон — проекция исходной 3D-разметки. Здесь проверяются визуальные решения по зонам,
не идентичность машин, частота инференса или управление автомобилем.</p></div>
<noscript><p class="error">Для просмотра нужен JavaScript; файлы и сеть дополнительно не используются.</p></noscript>
</main>
<script type="application/json" id="video-data">"""

DOCUMENT_END = """</script>
<script>
(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const data = JSON.parse(byId("video-data").textContent);
  const clipSelect = byId("clip");
  const modeSelect = byId("mode");
  const rateSelect = byId("rate");
  const slider = byId("frame");
  const playButton = byId("play");
  const previousButton = byId("prev");
  const nextButton = byId("next");
  const error = byId("error");
  const width = byId("left-image").width;
  const bands = 4;
  const svgNamespace = "http://www.w3.org/2000/svg";
  const labelById = new Map(data.modes.map(mode => [mode.id, mode.label]));
  const baseline = data.modes.findIndex(mode => mode.id === data.baseline);
  const stateLabels = { "-1": "Нет", "0": "Не уверен", "1": "Есть" };
  const reasonLabels = {
    raw: "текущий кадр",
    pending_confirmation: "ждём подтверждения",
    confirmed: "подтверждено",
    held_uncertain: "краткое удержание",
    score_ema: "сглажена оценка",
    box_ema: "сглажена рамка"
  };
  const resetLabels = {
    initial: "Начало клипа: история пуста.",
    gap: "Разрыв источника: история сброшена."
  };
  const imageCache = new Map();

  let clipIndex = 0;
  let frameIndex = 0;
  let playing = false;
  let timer = null;
  let generation = 0;

  function option(value, text) {
    const item = document.createElement("option");
    item.value = value;
    item.textContent = text;
    return item;
  }

  data.clips.forEach((clip, index) => {
    clipSelect.append(option(String(index), String(index + 1) + " · " + clip.clip));
  });
  data.modes.forEach(mode => modeSelect.append(option(mode.id, mode.label)));
  modeSelect.value = data.selected;
  byId("dev").textContent = "Режим по умолчанию выбран только на dev: " + labelById.get(data.selected) + ".";
  byId("totals").textContent = data.clips.length + " клипов · " +
    data.clips.reduce((count, clip) => count + clip.frames.length, 0) + " кадров · " +
    data.modes.length + " режимов";

  function clearTimer() {
    window.clearTimeout(timer);
    timer = null;
  }

  function setPlaying(value) {
    clearTimer();
    playing = value;
    playButton.setAttribute("aria-pressed", String(value));
    playButton.textContent = value ? "Пауза" : "Воспроизвести";
  }

  function imageFor(frame) {
    if (imageCache.has(frame.sample)) return imageCache.get(frame.sample);
    const pending = new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Не удалось открыть встроенный кадр."));
      image.src = frame.image;
    });
    imageCache.set(frame.sample, pending);
    if (imageCache.size > 6) imageCache.delete(imageCache.keys().next().value);
    return pending;
  }

  function svgShape(type, attrs, cssClass) {
    const item = document.createElementNS(svgNamespace, type);
    item.setAttribute("class", cssClass);
    Object.entries(attrs).forEach(([key, value]) => item.setAttribute(key, String(value)));
    return item;
  }

  function rect(box, cssClass) {
    return svgShape("rect", {
      x: width * box[0], y: width * box[1],
      width: width * (box[2] - box[0]), height: width * (box[3] - box[1])
    }, cssClass);
  }

  function reasonLabel(value) {
    return value.split("+").map(part => reasonLabels[part] || part).join(" · ");
  }

  function zoneCard(state, reason, index) {
    const card = document.createElement("li");
    card.className = "zone";
    card.dataset.state = String(state);
    const summary = document.createElement("b");
    summary.textContent = "Зона " + (index + 1) + " · " + stateLabels[state];
    const detail = document.createElement("span");
    detail.textContent = reasonLabel(reason);
    card.append(summary, detail);
    return card;
  }

  function drawPanel(side, source, frame, result) {
    const surface = byId(side + "-image");
    surface.getContext("2d").drawImage(source, 0, 0, width, width);
    surface.setAttribute("aria-label", "Кадр " + frame.index + ", передняя камера");
    const overlay = [];
    result.states.forEach((state, zone) => {
      if (state === 0) overlay.push(rect([zone / bands, 0, (zone + 1) / bands, 1], "shade"));
    });
    for (let zone = 1; zone < bands; ++zone) {
      const x = zone * width / bands;
      overlay.push(svgShape("line", { x1: x, y1: 0, x2: x, y2: width }, "guide"));
    }
    for (const box of frame.truth) if (box !== null) overlay.push(rect(box, "truth-box"));
    for (const box of result.boxes) if (box !== null) overlay.push(rect(box, "pred-box"));
    byId(side + "-overlay").replaceChildren(...overlay);
    byId(side + "-zones").replaceChildren(
      ...result.states.map((state, zone) => zoneCard(state, result.reasons[zone], zone))
    );
    byId(side + "-reset").textContent = resetLabels[result.reset] || "";
  }

  async function render() {
    clearTimer();
    const request = ++generation;
    const clip = data.clips[clipIndex];
    const frame = clip.frames[frameIndex];
    const right = data.modes.findIndex(mode => mode.id === modeSelect.value);
    slider.max = String(clip.frames.length - 1);
    slider.value = String(frameIndex);
    slider.setAttribute("aria-valuetext", "Кадр " + (frameIndex + 1) + " из " + clip.frames.length);
    previousButton.disabled = frameIndex === 0;
    nextButton.disabled = frameIndex + 1 === clip.frames.length;

    try {
      const image = await imageFor(frame);
      if (request !== generation) return;
      error.hidden = true;
      // Drawing both panes from one decoded image prevents comparing different frames while decoding.
      drawPanel("left", image, frame, frame.predictions[baseline]);
      drawPanel("right", image, frame, frame.predictions[right]);
      byId("right-title").textContent = data.modes[right].label;
      const seconds = (frame.offset_us / 1000000).toLocaleString("ru-RU", {
        minimumFractionDigits: 2, maximumFractionDigits: 2
      });
      byId("readout").textContent = "Кадр " + (frameIndex + 1) + "/" + clip.frames.length +
        " · +" + seconds + " с · исходный индекс " + frame.index;
    } catch (err) {
      if (request !== generation) return;
      setPlaying(false);
      error.textContent = err.message;
      error.hidden = false;
      return;
    }

    const next = clip.frames[frameIndex + 1];
    if (next) imageFor(next).catch(() => {});
    if (!playing) return;
    if (!next) {
      setPlaying(false);
      return;
    }
    const delayMs = (next.offset_us - frame.offset_us) / 1000 / Number(rateSelect.value);
    timer = window.setTimeout(() => {
      frameIndex += 1;
      void render();
    }, delayMs);
  }

  function jump(index) {
    setPlaying(false);
    frameIndex = Math.max(0, Math.min(index, data.clips[clipIndex].frames.length - 1));
    void render();
  }

  function toggle() {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (frameIndex + 1 === data.clips[clipIndex].frames.length) frameIndex = 0;
    setPlaying(true);
    void render();
  }

  clipSelect.addEventListener("change", () => {
    clipIndex = Number(clipSelect.value);
    jump(0);
  });
  modeSelect.addEventListener("change", render);
  rateSelect.addEventListener("change", render);
  slider.addEventListener("input", () => jump(Number(slider.value)));
  previousButton.addEventListener("click", () => jump(frameIndex - 1));
  nextButton.addEventListener("click", () => jump(frameIndex + 1));
  playButton.addEventListener("click", toggle);
  document.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.repeat ||
        (event.target instanceof HTMLElement &&
         (event.target.isContentEditable || event.target.closest("input,select,button,textarea,a")))) return;
    if (event.key === "ArrowLeft") jump(frameIndex - 1);
    else if (event.key === "ArrowRight") jump(frameIndex + 1);
    else if (event.key === " ") toggle();
    else return;
    event.preventDefault();
  });
  void render();
})();
</script>
</body>
</html>
"""
