// Sterowanie podglądem. Każda wielkość liczbowa na stronie pochodzi z kerneli GPU.
import { ParticleEngine, ALL, NTOK, H1 } from './gpu-engine.js';

const el = id => document.getElementById(id);
const FMT = new Intl.NumberFormat('pl-PL', { maximumFractionDigits: 3 });
const SCI = v => (Math.abs(v) > 0 && Math.abs(v) < 1e-3 ? v.toExponential(2) : FMT.format(v));
const C = { text: '#f2f6fa', muted: '#aebdca', panel: '#142131', grid: '#2b3e50', blue: '#62d4dd', orange: '#ffb967' };
const PALETTE = ['#62d4dd', '#ffb967', '#d88fe5', '#a9d776', '#8da6ff', '#f0909d', '#dfd889', '#b5cedd'];
const CONFIG = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['sendDataToCloud'] };
const MILESTONES = [0, 100, 500];
const COLORS = {
  milestone: [[0, 'Norma ruchu w 768D (zapis)'], [1, 'Prędkość na odcinku w rzucie PCA (GPU)'], [2, 'Przesunięcie |Δ| w rzucie PCA (GPU)']],
  control: [[3, 'Tożsamość patcha'], [0, 'Norma ruchu w 768D od startu (zapis)'], [1, 'Pojedynczy krok w 768D (zapis)'], [2, 'Przesunięcie |Δ| w rzucie PCA (GPU)']],
};
const COLOR_LABEL = { 0: 'norma 768D', 1: 'prędkość PCA / krok', 2: '|Δ| PCA', 3: 'patch' };

let META, engine;
const S = { t: 0, playing: false, dwell: 0, last: 0, dirty: true, curves: null, heat: null, medianCache: new Map(),
  lastStatStep: -1, lastStatTime: 0, statBusy: false, viewSeq: 0, frames: [] };

const mode = () => el('mode').value;
const kind = () => (mode() === 'exact' ? 'control' : 'milestone');
const selected = () => Number(el('image').value);

function viewImage() {
  if (mode() === 'global') return ALL;
  if (mode() === 'image') return selected();
  return selected() < 0 ? ALL : selected();
}

/** Czas -> parametry kernela. Zwraca też, czy to zapisany stan. */
function timeSpec() {
  if (kind() === 'control') { const step = Math.round(S.t); return { step, recorded: true, state: null }; }
  const t = S.t;
  if (t <= 100) return { ka: 0, kb: 1, s: t / 100, recorded: t === 0 || t === 100, state: t === 0 ? 0 : t === 100 ? 1 : null, step: Math.floor(t) };
  return { ka: 1, kb: 2, s: (t - 100) / 400, recorded: t === 500, state: t === 500 ? 2 : null, step: Math.floor(t) };
}
const lastRecordedState = t => (t >= 500 ? 2 : t >= 100 ? 1 : 0);

function fillColorSelect(reset = false) {
  const prev = Number(el('color').value);
  const list = COLORS[kind()];
  el('color').replaceChildren(...list.map(([v, label]) => { const o = document.createElement('option'); o.value = v; o.textContent = label; return o; }));
  el('color').value = !reset && list.some(([v]) => v === prev) ? prev : list[0][0];
}

function fillImages() {
  const m = mode();
  el('image-wrap').classList.toggle('hidden', m === 'global');
  if (m === 'exact') {
    const opts = [[-1, 'Wszystkie 16 obrazów · 128 trajektorii'], ...META.control_images.map((img, k) => [k, `${String(img).padStart(4, '0')} · ${META.images[img].frame_id}`])];
    el('image').replaceChildren(...opts.map(([v, label]) => { const o = document.createElement('option'); o.value = v; o.textContent = label; return o; }));
    el('image').value = -1;
  } else if (m === 'image') {
    el('image').replaceChildren(...META.images.map((r, i) => { const o = document.createElement('option'); o.value = i; o.textContent = `${String(i).padStart(4, '0')} · ${r.frame_id}`; return o; }));
    el('image').value = 100;
  }
  el('transit-wrap').classList.toggle('hidden', m === 'exact');
  el('trail-wrap').classList.toggle('hidden', m !== 'exact');
  el('paths-wrap').classList.toggle('hidden', m !== 'image');
}

function defaultsForMode() {
  const m = mode();
  if (m === 'global') { el('opacity').value = 0.065; el('size').value = 1.5; }
  else if (m === 'image') { el('opacity').value = 0.8; el('size').value = 4; }
  else { el('opacity').value = 1; el('size').value = 6; }
  engine.camera.alpha = Number(el('opacity').value);
  engine.camera.size = Number(el('size').value);
}

// ------------------------------------------------------------------ widok
async function applyView({ recurves = false } = {}) {
  const seq = ++S.viewSeq;
  stopStats();
  const k = kind(), image = viewImage();
  const fi = await engine.setView({ kind: k, image, movement: el('movement').checked, color: Number(el('color').value),
    trails: el('trails').checked, paths: mode() === 'image' && el('paths').checked });
  if (seq !== S.viewSeq) return;
  S.frameInfo = fi;
  if (recurves) {
    const sel = mode() === 'global' ? [0, 0] : mode() === 'image' ? [2, selected()] : selected() < 0 ? [1, 0] : [2, META.control_images[selected()]];
    S.curves = await engine.curves(...sel);
    S.heat = k === 'milestone' ? await engine.heat(image) : null;
    S.medianCache.clear();
    await drawCurves();
  }
  if (seq !== S.viewSeq) return;
  const count = engine.view.count;
  const total = mode() === 'global' ? 900000 : count;
  el('counts').textContent = `${FMT.format(count)} / ${FMT.format(total)} cząsteczek na GPU`;
  el('legend').textContent = `${el('movement').checked ? 'Δ ' : ''}PC1 / PC2 / PC3 · wspólna skala · kolor: ${COLOR_LABEL[engine.view.color]} (max ${SCI(fi.colorMax)})`;
  el('sampling-note').textContent = mode() === 'exact'
    ? 'Każdy punkt to rzeczywiście zapisany stan w tym kroku. Ślad pokazuje ostatnie 40 zapisanych kroków.'
    : mode() === 'global'
      ? 'Wszystkie 900 000 cząsteczek bez próbkowania. W krokach 0, 100 i 500 położenia są zapisane. Pomiędzy nimi GPU prowadzi je po prostych odcinkach.'
      : 'Wszystkie 600 cząsteczek tego obrazu. Zapisane stany to kroki 0, 100 i 500; pomiędzy nimi odcinek liniowy.';
  S.lastStatStep = -1;
  S.dirty = true;
  await updateStats(true);
}

// ------------------------------------------------------------------ wykresy
function commonLayout() {
  return { paper_bgcolor: C.panel, plot_bgcolor: C.panel, font: { family: 'system-ui,sans-serif', color: C.text, size: 12 },
    margin: { l: 48, r: 25, t: 20, b: 42 }, showlegend: true, legend: { orientation: 'h', y: 1.14 }, uirevision: 'same' };
}

async function drawCurves() {
  const x = Array.from({ length: H1 }, (_, i) => i), cv = S.curves;
  const conf = x.map(i => cv[i * 4]), iou = x.map(i => cv[i * 4 + 1]);
  const layout = commonLayout();
  layout.xaxis = { title: 'Rzeczywisty krok optymalizacji', range: [0, 500], gridcolor: C.grid };
  layout.yaxis = { title: 'Wartość', range: [0, 1.02], tickformat: '.0%', gridcolor: C.grid };
  layout.shapes = [{ type: 'line', x0: 0, x1: 500, y0: .9, y1: .9, line: { color: C.muted, dash: 'dot', width: 1 } },
    { type: 'line', x0: S.t, x1: S.t, y0: 0, y1: 1, line: { color: C.muted, width: 1 } }];
  await Plotly.react('quality', [
    { x, y: conf, type: 'scatter', mode: 'lines', name: 'Pewność', line: { color: C.blue, width: 2 } },
    { x, y: iou, type: 'scatter', mode: 'lines', name: 'IoU', line: { color: C.orange, width: 2, dash: 'dash' } }], layout, CONFIG);
  el('metric-title').textContent = mode() === 'global' ? 'Detekcja · średnia z 1500 obrazów (GPU)'
    : mode() === 'exact' && selected() < 0 ? 'Detekcja · średnia panelu 16 obrazów (GPU)' : 'Detekcja tego obrazu';
}

async function drawHeat(state) {
  const layout = commonLayout();
  layout.margin = { l: 35, r: 30, t: 5, b: 28 };
  layout.xaxis = { title: 'Kolumna patcha', dtick: 5 };
  layout.yaxis = { title: 'Wiersz', autorange: 'reversed', dtick: 5 };
  const z = [], h = S.heat;
  for (let r = 0; r < 20; r++) z.push(Array.from(h.subarray(state * NTOK + r * 30, state * NTOK + r * 30 + 30)));
  const zmax = Math.max(...h, 1e-6);
  await Plotly.react('heat', [{ z, type: 'heatmap', zmin: 0, zmax, colorscale: [[0, '#142131'], [.4, '#2e8a96'], [1, '#ffb967']],
    colorbar: { thickness: 10, title: 'L2' }, hovertemplate: 'Wiersz %{y} · kolumna %{x}<br>Ruch 768D: %{z:.4g}<extra></extra>' }], layout, CONFIG);
  el('heat-title').textContent = `Ruch w układzie patchy 20 × 30 · stan ${MILESTONES[state]}`;
  el('heat-note').textContent = mode() === 'global'
    ? 'Średnia norma 768D na pozycję patcha po 1500 obrazach, policzona kernelem heat. Skala stała dla trzech stanów.'
    : 'Norma 768D wszystkich 600 cząsteczek w ich oryginalnej siatce.';
}

async function drawBars(values) {
  const layout = commonLayout();
  layout.margin = { l: 48, r: 20, t: 5, b: 40 };
  layout.xaxis = { title: 'Numer śledzonego patcha' };
  layout.yaxis = { title: 'Norma ruchu 768D', range: [0, S.frameInfo.normMax * 1.05], gridcolor: C.grid };
  const tokens = META.control_tokens[0].map(String);
  await Plotly.react('heat', [{ x: tokens, y: Array.from(values), type: 'bar', marker: { color: PALETTE },
    hovertemplate: 'Patch %{x}<br>Ruch 768D: %{y:.4g}<extra></extra>' }], layout, CONFIG);
  el('heat-title').textContent = selected() < 0 ? 'Średni ruch panelu według patcha' : 'Ile poruszyło się 8 śledzonych cząsteczek';
  el('heat-note').textContent = 'Średnia norma 768D dla każdego patcha w tym kroku, policzona kernelem bars.';
}

// ------------------------------------------------------------------ statystyki zależne od czasu
function stopStats() { S.statBusy = false; }

async function updateStats(force = false) {
  const now = performance.now();
  const step = Math.floor(S.t);
  if (!force && (step === S.lastStatStep || now - S.lastStatTime < 120 || S.statBusy)) return;
  S.statBusy = true; S.lastStatStep = step; S.lastStatTime = now;
  const seq = S.viewSeq;
  try {
    const cv = S.curves;
    el('step-value').textContent = kind() === 'control' ? step : (timeSpec().recorded ? step : `${step} · odcinek`);
    if (cv) {
      el('conf').textContent = `Pewność: ${(100 * cv[step * 4]).toFixed(2)}%`;
      el('iou').textContent = `IoU: ${(100 * cv[step * 4 + 1]).toFixed(2)}%`;
      Plotly.relayout('quality', { 'shapes[1].x0': S.t, 'shapes[1].x1': S.t });
    }
    const ts = timeSpec();
    el('transit').textContent = kind() === 'milestone' && !ts.recorded
      ? `Odcinek liniowy ${MILESTONES[ts.ka]} → ${MILESTONES[ts.kb]} · stan niezapisany` : '';
    let med;
    if (kind() === 'milestone') {
      const st = ts.recorded ? ts.state : lastRecordedState(S.t);
      const key = `${viewImage()}:${st}`;
      if (!S.medianCache.has(key)) S.medianCache.set(key, await engine.orderStats({ source: 0, image: viewImage(), state: st }));
      med = S.medianCache.get(key);
      el('movement-norm').textContent = `Mediana ruchu 768D w kroku ${MILESTONES[st]}: ${SCI(med.median)} · max ${SCI(med.max)}`;
      if (S.heat && S.heatState !== st) { S.heatState = st; await drawHeat(st); }
    } else {
      med = await engine.orderStats({ source: 1, image: viewImage(), step });
      el('movement-norm').textContent = `Mediana ruchu 768D: ${SCI(med.median)} · max ${SCI(med.max)}`;
      await drawBars(await engine.bars(viewImage(), step));
    }
    if (seq !== S.viewSeq) return;
    const r = META.images[selected()] || {};
    const both = cv ? cv[step * 4 + 3] : 0;
    el('phase').textContent = mode() === 'global' ? `${both} / 1500 osiąga oba progi w kroku ${step}`
      : mode() === 'exact' && selected() < 0 ? `${both} / 16 osiąga oba progi w kroku ${step}`
        : (mode() === 'exact' ? META.images[META.control_images[selected()]] : r).first_success_step != null
          && step >= (mode() === 'exact' ? META.images[META.control_images[selected()]] : r).first_success_step
          ? 'Po pierwszym sukcesie · minimalizacja przesunięcia' : 'Dążenie do poprawnej detekcji';
  } finally { S.statBusy = false; }
}

// ------------------------------------------------------------------ osie
function drawAxes() {
  const svg = el('axes'), F = S.frameInfo;
  if (!F) return;
  const w = el('gpu').clientWidth, h = el('gpu').clientHeight;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const corners = Array.from({ length: 8 }, (_, j) => engine.projectToScreen([0, 1, 2].map(i => (j >> i) & 1 ? F.hi[i] : F.lo[i])));
  const d = el('movement').checked ? 'Δ ' : '';
  svg.innerHTML = [0, 1, 2].map(i => {
    const a = corners[0], b = corners[1 << i];
    return `<text x="${(a[0] + b[0]) / 2}" y="${(a[1] + b[1]) / 2 - 7}" fill="#aebdca" text-anchor="middle" font-size="12">${d}PC${i + 1}</text>`;
  }).join('');
  el('legend').title = F.lo.map((v, i) => `PC${i + 1}: ${v.toFixed(4)} … ${F.hi[i].toFixed(4)}`).join('\n');
}

// ------------------------------------------------------------------ pętla
function frame(now) {
  requestAnimationFrame(frame);
  if (!engine || !engine.view || engine.busy || document.hidden) { S.last = now; return; }
  const dt = Math.min(0.1, (now - (S.last || now)) / 1000);
  S.last = now;
  if (S.playing) advance(dt);
  if (el('spin').checked) { engine.camera.yaw += 0.5 * dt; S.dirty = true; }
  if (!S.dirty) return;
  S.dirty = false;
  engine.simulate(timeSpec());
  engine.draw();
  drawAxes();
  el('step').value = Math.round(S.t);
  S.frames.push(now);
  if (S.frames.length > 61) S.frames.shift();
  const fps = S.frames.length > 10 && now - S.frames[0] < 2000 ? 1000 * (S.frames.length - 1) / (now - S.frames[0]) : null;
  el('gpu-status').textContent = `WebGPU · ${engine.adapterInfo} · kernel ruchu + rysowanie ${FMT.format(engine.view.count)} cząsteczek${fps ? ` · ${fps.toFixed(0)} kl./s` : ''}`;
  updateStats();
}

function advance(dt) {
  const rate = Number(el('speed').value);
  if (kind() === 'milestone' && !el('transitions').checked) {
    S.dwell += dt;
    if (S.dwell < 1.2) return;
    S.dwell = 0;
    const next = MILESTONES.find(m => m > S.t);
    if (next == null) return stop();
    S.t = next; S.dirty = true; return;
  }
  if (S.dwell > 0) { S.dwell -= dt; return; }
  const before = S.t;
  S.t = Math.min(500, S.t + rate * dt);
  if (kind() === 'milestone' && before < 100 && S.t >= 100) { S.t = 100; S.dwell = 0.8; }
  S.dirty = true;
  if (S.t >= 500) stop();
}

function play() {
  if (S.playing) return stop();
  if (S.t >= 500) S.t = 0;
  S.playing = true; S.dwell = 0; el('play').textContent = 'Pauza';
}
function stop() { S.playing = false; el('play').textContent = 'Odtwórz'; S.lastStatStep = -1; updateStats(true); }

function setT(t) {
  if (kind() === 'milestone' && !el('transitions').checked) t = MILESTONES.reduce((a, b) => (Math.abs(b - t) < Math.abs(a - t) ? b : a));
  S.t = t; S.dirty = true;
}

// ------------------------------------------------------------------ sterowanie myszą
function installControls() {
  const canvas = el('gpu');
  let prev = null, travel = 0;
  canvas.addEventListener('pointerdown', e => { if (e.button !== 0) return; canvas.setPointerCapture(e.pointerId); prev = [e.clientX, e.clientY]; travel = 0; });
  canvas.addEventListener('pointermove', e => {
    if (!prev) return;
    const dx = e.clientX - prev[0], dy = e.clientY - prev[1];
    travel += Math.abs(dx) + Math.abs(dy);
    engine.camera.yaw += dx * 0.008;
    engine.camera.pitch = Math.max(-1.5, Math.min(1.5, engine.camera.pitch + dy * 0.008));
    prev = [e.clientX, e.clientY]; S.dirty = true;
  });
  canvas.addEventListener('pointerup', async e => {
    if (prev && travel < 4) {
      const r = canvas.getBoundingClientRect();
      const hit = await engine.pick(e.clientX - r.left, e.clientY - r.top);
      el('point-detail').textContent = hit ? describe(hit) : 'Brak cząsteczki w tym miejscu.';
    }
    prev = null;
  });
  canvas.addEventListener('pointercancel', () => { prev = null; });
  canvas.addEventListener('wheel', e => { e.preventDefault(); engine.camera.zoom = Math.max(0.2, Math.min(40, engine.camera.zoom * Math.exp(-e.deltaY * 0.001))); S.dirty = true; }, { passive: false });
  new ResizeObserver(() => { S.dirty = true; }).observe(canvas);
}

function describe(hit) {
  let image, token;
  if (mode() === 'global') { image = Math.floor(hit.id / 600); token = hit.id % 600; }
  else if (mode() === 'image') { image = selected(); token = hit.id; }
  else {
    const slot = selected() < 0 ? Math.floor(hit.id / 8) : selected();
    image = META.control_images[slot]; token = META.control_tokens[slot][hit.id % 8];
  }
  const c = engine.view.color === 3 ? '' : ` · ${COLOR_LABEL[engine.view.color]}: ${SCI(hit.color)}`;
  return `Obraz ${image} · patch ${token} (wiersz ${Math.floor(token / 30)}, kolumna ${token % 30})${c} · pozycja ${hit.xyz.map(v => v.toFixed(3)).join(', ')}`;
}

// ------------------------------------------------------------------ start
async function init() {
  try {
    const r = await fetch('/api/meta');
    if (!r.ok) throw new Error(await r.text());
    META = await r.json();
    el('loading').textContent = 'Kopiowanie surowych tablic do pamięci GPU i kompilacja kerneli WGSL…';
    engine = await ParticleEngine.create(el('gpu'), META, f => { el('loading').textContent = `Ładowanie danych do GPU: ${(100 * f).toFixed(0)}%`; });
    engine.onLost = info => { el('loading').textContent = `Utracono urządzenie GPU: ${info.message}. Odśwież stronę.`; };
    const v = META.pca_variance;
    el('variance').textContent = `PCA 3D: ${(100 * v.reduce((a, b) => a + b, 0)).toFixed(2)}% wariancji próbki`;
    fillImages(); fillColorSelect(); defaultsForMode(); installControls();
    el('mode').onchange = async () => { stop(); fillImages(); fillColorSelect(true); defaultsForMode(); S.t = 0; S.heatState = null; await applyView({ recurves: true }); };
    el('image').onchange = async () => { S.heatState = null; await applyView({ recurves: true }); };
    for (const id of ['movement', 'color', 'trails', 'paths']) el(id).onchange = () => applyView();
    el('transitions').onchange = () => setT(S.t);
    el('step').oninput = () => { S.playing = false; el('play').textContent = 'Odtwórz'; setT(Number(el('step').value)); };
    el('play').onclick = play;
    el('begin').onclick = () => { stop(); setT(0); };
    el('opacity').oninput = () => { engine.camera.alpha = Number(el('opacity').value); S.dirty = true; };
    el('size').oninput = () => { engine.camera.size = Number(el('size').value); S.dirty = true; };
    el('spin').onchange = () => { S.frames = []; S.dirty = true; };
    el('verify').onclick = runVerify;
    await applyView({ recurves: true });
    el('loading').textContent = '';
    el('snapshot').onclick = () => saveSnapshot().catch(e => { el('point-detail').textContent = e.message; });
    window.__particles = { engine, S, META, applyView, setT, saveSnapshot, timeSpec, frame, updateStats };
    requestAnimationFrame(frame);
    if (new URLSearchParams(location.search).has('selftest')) runVerify();
  } catch (e) {
    console.error(e);
    el('loading').textContent = `Nie udało się uruchomić podglądu GPU: ${e.message}`;
  }
}

async function saveSnapshot(name) {
  const ts = timeSpec();
  if (!name) name = `${mode()}_${mode() === 'global' ? 'zbior' : `obraz${selected()}`}_krok${Math.round(S.t)}${el('movement').checked ? '_delta' : ''}`;
  engine.simulate(ts);
  const blob = await engine.snapshot();
  const r = await fetch(`/api/snapshot?name=${encodeURIComponent(name)}`, { method: 'POST', body: blob });
  if (!r.ok) throw new Error('Serwer nie zapisał klatki');
  const res = await r.json();
  el('point-detail').textContent = `Zapisano ${res.saved} w folderze wyników.`;
  return res;
}

async function runVerify() {
  stop();
  el('verify').disabled = true;
  const out = el('verify-out');
  out.textContent = 'Uruchamiam kernele na przypadkach testowych…';
  try {
    const { runSelfTest } = await import('./selftest.js');
    const report = await runSelfTest(engine, META, msg => { out.textContent = msg; });
    out.innerHTML = `<div class="${report.status === 'passed' ? 'ok' : 'bad'}"><strong>${report.cases_passed} / ${report.cases_total} przypadków zgodnych z numpy</strong> · zapisano WERYFIKACJA_GPU.json</div>`
      + report.cases.map(c => `<div class="${c.passed ? 'ok' : 'bad'}">${c.passed ? '✓' : '✗'} ${c.name} · ${c.compare || c.reason || ''}${c.max_abs_error != null ? ` · max błąd ${c.max_abs_error.toExponential(1)}` : ''}</div>`).join('');
    window.__verifyReport = report;
  } catch (e) {
    console.error(e);
    out.innerHTML = `<div class="bad">Weryfikacja przerwana: ${e.message}</div>`;
  } finally {
    el('verify').disabled = false;
    await applyView({ recurves: true });
  }
}

init();
