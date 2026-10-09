// Silnik WebGPU: compute shadery liczą ruch i statystyki, render rysuje wynik.
import { COMPUTE, RENDER } from './shaders.js';

export const ALL = 0xffffffff;
export const NIMG = 1500, NTOK = 600, H1 = 501, TRAIL_LEN = 40;
const DATA_NAMES = ['positions', 'movement_norm', 'control_positions', 'control_norm',
  'control_step_norm', 'confidence', 'iou', 'control_images'];

export class ParticleEngine {
  static async create(canvas, meta, progress = () => {}) {
    if (!navigator.gpu) throw new Error('Ta przeglądarka nie udostępnia WebGPU. Użyj Chrome lub Edge 113+.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('Brak adaptera WebGPU.');
    const need = { maxStorageBuffersPerShaderStage: 16 };
    if (adapter.limits.maxStorageBuffersPerShaderStage < 16) throw new Error('GPU obsługuje za mało buforów storage na etap shadera.');
    const device = await adapter.requestDevice({
      requiredLimits: {
        ...need,
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 << 20),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 256 << 20),
      },
    });
    const engine = new ParticleEngine(canvas, device, adapter, meta);
    await engine.init(progress);
    return engine;
  }

  constructor(canvas, device, adapter, meta) {
    this.canvas = canvas; this.device = device; this.meta = meta;
    this.adapterInfo = adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description || ''}`.trim() : 'WebGPU';
    this.context = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.lost = false;
    device.lost.then(info => { this.lost = true; this.onLost?.(info); });
    this.view = null;
    this.camera = { yaw: 0.7, pitch: -0.42, zoom: 1, size: 1.5, alpha: 0.065 };
  }

  buffer(size, usage, label) { return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage, label }); }

  async init(progress) {
    const d = this.device, S = GPUBufferUsage;
    // 1. Surowe tablice zapisu -> bufory storage (bez żadnego przeliczania na CPU).
    this.data = {};
    let loaded = 0;
    const total = DATA_NAMES.reduce((s, n) => s + this.meta.arrays[n].bytes, 0);
    await Promise.all(DATA_NAMES.map(async name => {
      const r = await fetch(`/api/array/${name}`);
      if (!r.ok) throw new Error(`Nie udało się pobrać ${name}`);
      const raw = await r.arrayBuffer();
      if (raw.byteLength !== this.meta.arrays[name].bytes) throw new Error(`Zły rozmiar ${name}`);
      const digest = await sha256(raw);
      if (digest !== this.meta.arrays[name].sha256) throw new Error(`Suma SHA-256 ${name} nie zgadza się`);
      const b = this.buffer(raw.byteLength, S.STORAGE | S.COPY_DST, name);
      d.queue.writeBuffer(b, 0, raw);
      this.data[name] = b;
      loaded += raw.byteLength;
      progress(loaded / total);
    }));
    this.uploadedBytes = total;

    // 2. Bufory wyników kerneli.
    this.out = this.buffer(NIMG * NTOK * 16, S.STORAGE | S.COPY_SRC, 'OUT');
    this.stats = this.buffer(8 * 4, S.STORAGE | S.COPY_DST | S.COPY_SRC, 'STATS');
    this.frame = this.buffer(5 * 16, S.STORAGE | S.COPY_SRC, 'FRAME');
    this.heatBuf = this.buffer(3 * NTOK * 4, S.STORAGE | S.COPY_SRC, 'HEAT');
    this.hist = this.buffer((65537 + 65536) * 4, S.STORAGE | S.COPY_DST | S.COPY_SRC, 'HIST');
    this.curv = this.buffer(H1 * 16, S.STORAGE | S.COPY_SRC, 'CURV');
    this.barsBuf = this.buffer(8 * 4, S.STORAGE | S.COPY_SRC, 'BARS');
    this.trail = this.buffer(Math.max(128 * (TRAIL_LEN + 1), NTOK * 3) * 16, S.STORAGE | S.COPY_SRC, 'TRAIL');
    this.params = this.buffer(64, S.UNIFORM | S.COPY_DST, 'PARAMS');
    this.camBuf = this.buffer(48, S.UNIFORM | S.COPY_DST, 'CAM');

    // 3. Potoki obliczeniowe z jawnym, wspólnym układem wiązań.
    const C = GPUShaderStage.COMPUTE;
    const ro = { buffer: { type: 'read-only-storage' } }, rw = { buffer: { type: 'storage' } };
    const layout0 = d.createBindGroupLayout({ entries: DATA_NAMES.map((_, i) => ({ binding: i, visibility: C, ...ro })) });
    const layout1 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: C, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4, 5, 6, 7, 8].map(i => ({ binding: i, visibility: C, ...rw }))] });
    const computeModule = d.createShaderModule({ code: COMPUTE, label: 'kernele ruchu' });
    await checkModule(computeModule, 'compute');
    const pl = d.createPipelineLayout({ bindGroupLayouts: [layout0, layout1] });
    this.kernels = {};
    for (const entry of ['kin_milestone', 'kin_paths', 'kin_control', 'kin_trail', 'stats_milestone', 'stats_control',
      'finalize_stats', 'heat', 'hist_hi', 'hist_lo', 'curves', 'bars']) {
      this.kernels[entry] = await d.createComputePipelineAsync({ layout: pl, compute: { module: computeModule, entryPoint: entry } });
    }
    this.group0 = d.createBindGroup({ layout: layout0, entries: DATA_NAMES.map((n, i) => ({ binding: i, resource: { buffer: this.data[n] } })) });
    this.group1 = d.createBindGroup({ layout: layout1, entries: [this.params, this.out, this.stats, this.frame, this.heatBuf,
      this.hist, this.curv, this.barsBuf, this.trail].map((b, i) => ({ binding: i, resource: { buffer: b } })) });

    // 4. Potoki rysowania.
    const V = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    const rlayout = d.createBindGroupLayout({ entries: [
      { binding: 0, visibility: V, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, ...ro },
      { binding: 2, visibility: GPUShaderStage.VERTEX, ...ro },
      { binding: 3, visibility: GPUShaderStage.VERTEX, ...ro }] });
    const renderModule = d.createShaderModule({ code: RENDER, label: 'render' });
    await checkModule(renderModule, 'render');
    const rpl = d.createPipelineLayout({ bindGroupLayouts: [rlayout] });
    const blend = { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    const mk = (vs, fs, topology, format, withBlend) => d.createRenderPipelineAsync({
      layout: rpl, primitive: { topology },
      vertex: { module: renderModule, entryPoint: vs },
      fragment: { module: renderModule, entryPoint: fs, targets: [withBlend ? { format, blend } : { format }] },
    });
    [this.pPoints, this.pLines, this.pBox, this.pPick] = await Promise.all([
      mk('vs_points', 'fs_points', 'triangle-list', this.format, true),
      mk('vs_lines', 'fs_lines', 'line-list', this.format, true),
      mk('vs_box', 'fs_lines', 'line-list', this.format, true),
      mk('vs_points', 'fs_pick', 'triangle-list', 'r32uint', false)]);
    this.rgroup = d.createBindGroup({ layout: rlayout, entries: [this.camBuf, this.out, this.frame, this.trail]
      .map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  }

  // ------------------------------------------------------------- parametry
  writeParams(p) {
    const u = new Uint32Array(16), f = new Float32Array(u.buffer);
    u[0] = p.count ?? 0; u[1] = p.image ?? ALL; u[2] = p.ka ?? 0; u[3] = p.kb ?? 0;
    f[4] = p.s ?? 0; u[5] = p.movement ? 1 : 0; u[6] = p.color ?? 0; u[7] = p.step ?? 0;
    u[8] = p.state ?? 0; u[9] = p.prefix ?? 0; u[10] = p.selKind ?? 0; u[11] = p.selImage ?? 0;
    u[12] = p.source ?? 0;
    this.device.queue.writeBuffer(this.params, 0, u);
  }

  dispatch(encoder, kernel, groups) {
    const pass = encoder.beginComputePass({ label: kernel });
    pass.setPipeline(this.kernels[kernel]);
    pass.setBindGroup(0, this.group0);
    pass.setBindGroup(1, this.group1);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  run(params, kernels) {
    this.writeParams(params);
    const enc = this.device.createCommandEncoder();
    for (const [k, groups] of kernels) this.dispatch(enc, k, groups);
    this.device.queue.submit([enc.finish()]);
  }

  async read(buffer, offset, size) {
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, offset, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap(); staging.destroy();
    return copy;
  }

  // ------------------------------------------------------------- widok
  /** view: {kind:'milestone'|'control', image (ALL lub nr obrazu / slot panelu), movement, color} */
  async setView(view) {
    this.view = { ...view };
    const v = this.view;
    if (v.kind === 'milestone') v.count = v.image === ALL ? NIMG * NTOK : NTOK;
    else v.count = v.image === ALL ? 128 : 8;
    this.device.queue.writeBuffer(this.stats, 0, new Uint32Array(8));
    const items = v.kind === 'milestone' ? v.count * 3 : v.count * H1;
    this.run({ count: v.count, image: v.image, movement: v.movement, color: v.color },
      [[v.kind === 'milestone' ? 'stats_milestone' : 'stats_control', Math.ceil(items / 256)], ['finalize_stats', 1]]);
    const f = new Float32Array(await this.read(this.frame, 0, 80));
    this.frameInfo = {
      lo: [f[0], f[1], f[2]], hi: [f[4], f[5], f[6]], center: [f[8], f[9], f[10]], radius: f[11],
      rawLo: [f[12], f[13], f[14]], normMax: f[15], rawHi: [f[16], f[17], f[18]], colorMax: f[19],
    };
    return this.frameInfo;
  }

  /** Jedna klatka ruchu: kernel kinematyki (+ ślady) i rysowanie. t: {ka,kb,s} albo {step}. */
  simulate(t) {
    const v = this.view, enc = this.device.createCommandEncoder();
    if (v.kind === 'milestone') {
      this.writeParams({ count: v.count, image: v.image, ka: t.ka, kb: t.kb, s: t.s, movement: v.movement, color: v.color });
      this.dispatch(enc, 'kin_milestone', Math.ceil(v.count / 256));
      if (v.paths) this.dispatch(enc, 'kin_paths', Math.ceil(v.count * 3 / 256));
    } else {
      this.writeParams({ count: v.count, image: v.image, step: t.step, movement: v.movement, color: v.color });
      this.dispatch(enc, 'kin_control', Math.ceil(v.count / 64));
      if (v.trails) this.dispatch(enc, 'kin_trail', Math.ceil(v.count * (TRAIL_LEN + 1) / 64));
    }
    this.device.queue.submit([enc.finish()]);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr)), h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.dpr = dpr;
  }

  writeCamera(lineLen = 0, lineIdentity = 0, w = this.canvas.width, h = this.canvas.height, scale = this.dpr) {
    const c = this.camera, v = this.view;
    const f = new Float32Array([c.yaw, c.pitch, c.zoom, w / h,
      c.size * scale, w, h, c.alpha,
      v.color, lineLen, v.kind === 'control' ? 0.9 : 0.35, lineIdentity]);
    this.device.queue.writeBuffer(this.camBuf, 0, f);
  }

  /** Rysuje scenę do dowolnego celu: płótna strony albo tekstury poza ekranem. */
  encodeScene(target, w, h, scale) {
    const v = this.view;
    const lines = v.kind === 'control' && v.trails ? { len: TRAIL_LEN + 1, n: v.count, id: 1 }
      : v.kind === 'milestone' && v.paths ? { len: 3, n: v.count, id: 0 } : null;
    this.writeCamera(lines ? lines.len : 0, lines ? lines.id : 0, w, h, scale);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: target,
      clearValue: { r: 0.078, g: 0.129, b: 0.192, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setBindGroup(0, this.rgroup);
    pass.setPipeline(this.pBox); pass.draw(24);
    if (lines) { pass.setPipeline(this.pLines); pass.draw(2 * (lines.len - 1), lines.n); }
    pass.setPipeline(this.pPoints); pass.draw(6, v.count);
    pass.end();
    return enc;
  }

  draw() {
    if (!this.view || this.lost) return;
    this.resize();
    const enc = this.encodeScene(this.context.getCurrentTexture().createView(), this.canvas.width, this.canvas.height, this.dpr);
    this.device.queue.submit([enc.finish()]);
  }

  /** Klatka poza ekranem jako PNG (działa także w karcie w tle). */
  async snapshot(width = 1600, height = 1000, scale = 1.5) {
    const bpr = Math.ceil(width * 4 / 256) * 256;
    const tex = this.device.createTexture({ size: [width, height], format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const staging = this.device.createBuffer({ size: bpr * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.encodeScene(tex.createView(), width, height, scale);
    enc.copyTextureToBuffer({ texture: tex }, { buffer: staging, bytesPerRow: bpr }, [width, height]);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(staging.getMappedRange());
    const img = new ImageData(width, height);
    const bgra = this.format === 'bgra8unorm';
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const s = y * bpr + x * 4, d = (y * width + x) * 4;
        img.data[d] = src[s + (bgra ? 2 : 0)]; img.data[d + 1] = src[s + 1]; img.data[d + 2] = src[s + (bgra ? 0 : 2)]; img.data[d + 3] = 255;
      }
    }
    staging.unmap(); staging.destroy(); tex.destroy();
    const canvas = new OffscreenCanvas(width, height);
    canvas.getContext('2d').putImageData(img, 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
  }

  /** Wybór punktu: osobny przebieg rysuje identyfikatory do tekstury r32uint. */
  async pick(cssX, cssY) {
    this.resize();
    const w = this.canvas.width, h = this.canvas.height, v = this.view;
    const x = Math.min(w - 1, Math.max(0, Math.floor(cssX * w / this.canvas.clientWidth)));
    const y = Math.min(h - 1, Math.max(0, Math.floor(cssY * h / this.canvas.clientHeight)));
    this.writeCamera();
    const tex = this.device.createTexture({ size: [w, h], format: 'r32uint', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const staging = this.device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
    pass.setBindGroup(0, this.rgroup); pass.setPipeline(this.pPick); pass.draw(6, v.count); pass.end();
    enc.copyTextureToBuffer({ texture: tex, origin: [x, y] }, { buffer: staging, bytesPerRow: 256 }, [1, 1]);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const id = new Uint32Array(staging.getMappedRange())[0] - 1;
    staging.unmap(); staging.destroy(); tex.destroy();
    if (id < 0 || id >= v.count) return null;
    const value = new Float32Array(await this.read(this.out, id * 16, 16));
    return { id, xyz: [value[0], value[1], value[2]], color: value[3] };
  }

  /** Rzut punktu na ekran (CSS px) tym samym wzorem co shader; do testu wyboru. */
  projectToScreen(p) {
    const c = this.camera, F = this.frameInfo;
    let q = p.map((v, i) => (v - F.center[i]) / Math.max(F.radius, 1e-6));
    const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
    q = [cy * q[0] + sy * q[2], q[1], -sy * q[0] + cy * q[2]];
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
    q = [q[0], cp * q[1] - sp * q[2], sp * q[1] + cp * q[2]];
    const aspect = this.canvas.width / this.canvas.height;
    const nx = q[0] * c.zoom / aspect, ny = q[1] * c.zoom;
    return [(nx + 1) / 2 * this.canvas.clientWidth, (1 - ny) / 2 * this.canvas.clientHeight];
  }

  // ------------------------------------------------------------- statystyki
  /** Mapa 20x30 dla trzech zapisanych stanów: średnia po 1500 obrazach lub jeden obraz. */
  async heat(image) {
    this.run({ image }, [['heat', 3 * NTOK]]);
    return new Float32Array(await this.read(this.heatBuf, 0, 3 * NTOK * 4));
  }

  /** Dokładna mediana i maksimum normy 768D: dwa przebiegi histogramu po bitach na GPU. */
  async orderStats(sel) {
    const count = sel.source === 0 ? (sel.image === ALL ? NIMG * NTOK : NTOK) : (sel.image === ALL ? 128 : 8);
    const base = { count, image: sel.image, state: sel.state ?? 0, step: sel.step ?? 0, source: sel.source };
    const groups = Math.ceil(count / 256);
    this.device.queue.writeBuffer(this.hist, 0, new Uint32Array(65537 + 65536));
    this.run(base, [['hist_hi', groups]]);
    const hi = new Uint32Array(await this.read(this.hist, 0, 65537 * 4));
    const ranks = [Math.floor((count - 1) / 2), Math.floor(count / 2)];
    const values = [];
    for (const r of ranks) {
      let acc = 0, prefix = 0;
      while (acc + hi[prefix] <= r) { acc += hi[prefix]; prefix++; }
      this.device.queue.writeBuffer(this.hist, 65537 * 4, new Uint32Array(65536));
      this.run({ ...base, prefix }, [['hist_lo', groups]]);
      const lo = new Uint32Array(await this.read(this.hist, 65537 * 4, 65536 * 4));
      let low = 0;
      while (acc + lo[low] <= r) { acc += lo[low]; low++; }
      values.push(bitsToFloat((prefix << 16 | low) >>> 0));
    }
    const max = bitsToFloat(hi[65536]);
    return { n: count, low: values[0], high: values[1], median: Math.fround((Math.fround(values[0] + values[1])) / 2), max };
  }

  /** Krzywe 501 kroków: [średnia pewność, średnie IoU, ułamek obu progów, liczba obu progów]. */
  async curves(selKind, image = 0) {
    this.run({ selKind, selImage: image }, [['curves', H1]]);
    return new Float32Array(await this.read(this.curv, 0, H1 * 16));
  }

  async bars(panel, step) {
    this.run({ image: panel, step }, [['bars', 1]]);
    return new Float32Array(await this.read(this.barsBuf, 0, 32));
  }

  async readOut(count) { return this.read(this.out, 0, count * 16); }
  async readTrail(n) { return this.read(this.trail, 0, n * 16); }
  async idle() { await this.device.queue.onSubmittedWorkDone(); }
}

function bitsToFloat(bits) { const u = new Uint32Array([bits]); return new Float32Array(u.buffer)[0]; }

export async function sha256(buffer) {
  const h = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, '0')).join('');
}

async function checkModule(module, label) {
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === 'error');
  if (errors.length) throw new Error(`Błąd kompilacji WGSL (${label}): ` + errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | '));
}
