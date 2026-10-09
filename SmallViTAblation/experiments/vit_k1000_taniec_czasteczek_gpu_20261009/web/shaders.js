// Kernele WGSL: cały ruch i wszystkie statystyki cząsteczek liczy GPU.
// Dane wejściowe to surowe tablice projection.npz w ich oryginalnym układzie C:
//   P  positions          (1500, 3, 600, 3)  obraz, stan 0/100/500, patch, oś PCA
//   NM movement_norm      (1500, 3, 600)     norma L2 zmiany w pełnych 768D (zapis)
//   CP control_positions  (16, 501, 8, 3)    panel kontrolny, każdy krok
//   CN control_norm       (16, 501, 8)       norma 768D względem kroku 0 (zapis)
//   CS control_step_norm  (16, 501, 8)       norma 768D względem poprzedniego kroku (zapis)
//   CONF, IOU             (1500, 501)
//   CIMG control_images   (16)

export const COMMON = /* wgsl */`
const ALL: u32 = 0xffffffffu;
const NTOK: u32 = 600u;
const NIMG: u32 = 1500u;
const H1: u32 = 501u;
const TRAIL_LEN: u32 = 40u;
const STEPS = array<f32, 3>(0.0, 100.0, 500.0);

struct Params {
  count: u32, image: u32, ka: u32, kb: u32,
  s: f32, movement: u32, color: u32, step: u32,
  state: u32, prefix: u32, selKind: u32, selImage: u32,
  source: u32, pad0: u32, pad1: u32, pad2: u32,
};

@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read> NM: array<f32>;
@group(0) @binding(2) var<storage, read> CP: array<f32>;
@group(0) @binding(3) var<storage, read> CN: array<f32>;
@group(0) @binding(4) var<storage, read> CS: array<f32>;
@group(0) @binding(5) var<storage, read> CONF: array<f32>;
@group(0) @binding(6) var<storage, read> IOU: array<f32>;
@group(0) @binding(7) var<storage, read> CIMG: array<u32>;

@group(1) @binding(0) var<uniform> prm: Params;
@group(1) @binding(1) var<storage, read_write> OUT: array<vec4f>;
@group(1) @binding(2) var<storage, read_write> STATS: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> FRAME: array<vec4f>;
@group(1) @binding(4) var<storage, read_write> HEAT: array<f32>;
@group(1) @binding(5) var<storage, read_write> HIST: array<atomic<u32>>;
@group(1) @binding(6) var<storage, read_write> CURV: array<f32>;
@group(1) @binding(7) var<storage, read_write> BARS: array<f32>;
@group(1) @binding(8) var<storage, read_write> TRAIL: array<vec4f>;

fn pos(i: u32, k: u32, t: u32) -> vec3f {
  let b = ((i * 3u + k) * NTOK + t) * 3u;
  return vec3f(P[b], P[b + 1u], P[b + 2u]);
}
fn nrm(i: u32, k: u32, t: u32) -> f32 { return NM[(i * 3u + k) * NTOK + t]; }
fn cidx(sl: u32, st: u32, j: u32) -> u32 { return (sl * H1 + st) * 8u + j; }
fn cpos(sl: u32, st: u32, j: u32) -> vec3f {
  let b = cidx(sl, st, j) * 3u;
  return vec3f(CP[b], CP[b + 1u], CP[b + 2u]);
}
// Rozkład indeksu cząsteczki widoku: cały zbiór (obraz*600+patch) albo jeden obraz.
fn particle(p: u32) -> vec2u {
  if (prm.image == ALL) { return vec2u(p / NTOK, p % NTOK); }
  return vec2u(prm.image, p);
}
fn slotOf(q: u32) -> vec2u {
  if (prm.image == ALL) { return vec2u(q / 8u, q % 8u); }
  return vec2u(prm.image, q % 8u);
}
// Monotoniczne kodowanie f32 -> u32: porządek liczb = porządek kodów (dla atomicMax).
fn ord(f: f32) -> u32 {
  let u = bitcast<u32>(f);
  if ((u & 0x80000000u) != 0u) { return ~u; }
  return u | 0x80000000u;
}
fn unord(u: u32) -> f32 {
  if ((u & 0x80000000u) != 0u) { return bitcast<f32>(u & 0x7fffffffu); }
  return bitcast<f32>(~u);
}
`;

export const COMPUTE = COMMON + /* wgsl */`
// ---------------------------------------------------------------------------
// Kinematyka całej chmury / jednego obrazu: położenie w czasie, przesunięcie
// względem stanu 0 i wielkość koloru. W zapisanych stanach (s=0 lub s=1)
// wynik jest bitowo równy zapisowi; pomiędzy nimi to jawny odcinek liniowy.
@compute @workgroup_size(256)
fn kin_milestone(@builtin(global_invocation_id) g: vec3u) {
  let p = g.x;
  if (p >= prm.count) { return; }
  let it = particle(p);
  let a = pos(it.x, prm.ka, it.y);
  let b = pos(it.x, prm.kb, it.y);
  let p0 = pos(it.x, 0u, it.y);
  var x: vec3f;
  if (prm.s <= 0.0) { x = a; } else if (prm.s >= 1.0) { x = b; } else { x = a + (b - a) * prm.s; }
  let d = x - p0;
  var c: f32;
  if (prm.color == 0u) {
    let na = nrm(it.x, prm.ka, it.y);
    let nb = nrm(it.x, prm.kb, it.y);
    if (prm.s <= 0.0) { c = na; } else if (prm.s >= 1.0) { c = nb; } else { c = na + (nb - na) * prm.s; }
  } else if (prm.color == 1u) {
    c = length(b - a) / (STEPS[prm.kb] - STEPS[prm.ka]);
  } else {
    c = length(d);
  }
  var o = x;
  if (prm.movement == 1u) { o = d; }
  OUT[p] = vec4f(o, c);
}

// Odcinki przemieszczeń 0 -> 100 -> 500 dla widoku jednego obrazu.
@compute @workgroup_size(256)
fn kin_paths(@builtin(global_invocation_id) g: vec3u) {
  let e = g.x;
  if (e >= prm.count * 3u) { return; }
  let it = particle(e / 3u);
  let k = e % 3u;
  var x = pos(it.x, k, it.y);
  if (prm.movement == 1u) { x = x - pos(it.x, 0u, it.y); }
  TRAIL[e] = vec4f(x, f32(k) * 0.5);
}

// Panel kontrolny: rzeczywisty stan w każdym kroku 0..500, bez interpolacji.
@compute @workgroup_size(64)
fn kin_control(@builtin(global_invocation_id) g: vec3u) {
  let q = g.x;
  if (q >= prm.count) { return; }
  let sj = slotOf(q);
  let x = cpos(sj.x, prm.step, sj.y);
  let d = x - cpos(sj.x, 0u, sj.y);
  var c: f32;
  switch prm.color {
    case 0u: { c = CN[cidx(sj.x, prm.step, sj.y)]; }
    case 1u: { c = CS[cidx(sj.x, prm.step, sj.y)]; }
    case 2u: { c = length(d); }
    default: { c = f32(sj.y); }
  }
  var o = x;
  if (prm.movement == 1u) { o = d; }
  OUT[q] = vec4f(o, c);
}

// Ślad ostatnich 40 rzeczywistych kroków każdej trajektorii panelu.
@compute @workgroup_size(64)
fn kin_trail(@builtin(global_invocation_id) g: vec3u) {
  let e = g.x;
  if (e >= prm.count * (TRAIL_LEN + 1u)) { return; }
  let q = e / (TRAIL_LEN + 1u);
  let h = e % (TRAIL_LEN + 1u);
  let sj = slotOf(q);
  let st = u32(max(i32(prm.step) - i32(TRAIL_LEN) + i32(h), 0));
  var x = cpos(sj.x, st, sj.y);
  if (prm.movement == 1u) { x = x - cpos(sj.x, 0u, sj.y); }
  TRAIL[e] = vec4f(x, f32(h) / f32(TRAIL_LEN));
}

// ---------------------------------------------------------------------------
// Statystyki sceny po wszystkich zapisanych stanach widoku: granice osi
// (stałe w czasie), maksimum normy 768D i maksimum wielkości koloru.
// STATS: 0..2 = max(~ord(min)), 3..5 = max(ord(max)), 6 = norma, 7 = kolor.
var<workgroup> wred: array<atomic<u32>, 8>;

fn reduceStats(x: vec3f, normv: f32, colorv: f32) {
  atomicMax(&wred[0], ~ord(x.x));
  atomicMax(&wred[1], ~ord(x.y));
  atomicMax(&wred[2], ~ord(x.z));
  atomicMax(&wred[3], ord(x.x));
  atomicMax(&wred[4], ord(x.y));
  atomicMax(&wred[5], ord(x.z));
  atomicMax(&wred[6], ord(normv));
  atomicMax(&wred[7], ord(colorv));
}
fn flushStats(li: u32) {
  workgroupBarrier();
  if (li < 8u) {
    let v = atomicLoad(&wred[li]);
    if (v != 0u) { atomicMax(&STATS[li], v); }
  }
}

@compute @workgroup_size(256)
fn stats_milestone(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let e = g.x;
  if (e < prm.count * 3u) {
    let it = particle(e / 3u);
    let k = e % 3u;
    let p0 = pos(it.x, 0u, it.y);
    let pk = pos(it.x, k, it.y);
    var x = pk;
    if (prm.movement == 1u) { x = pk - p0; }
    let n = nrm(it.x, k, it.y);
    var c = n;
    if (prm.color == 1u) {
      let k1 = max(k, 1u);
      c = length(pos(it.x, k1, it.y) - pos(it.x, k1 - 1u, it.y)) / (STEPS[k1] - STEPS[k1 - 1u]);
    } else if (prm.color == 2u) {
      c = length(pk - p0);
    }
    reduceStats(x, n, c);
  }
  flushStats(li);
}

@compute @workgroup_size(256)
fn stats_control(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let e = g.x;
  if (e < prm.count * H1) {
    let sj = slotOf(e / H1);
    let st = e % H1;
    let pk = cpos(sj.x, st, sj.y);
    let d = pk - cpos(sj.x, 0u, sj.y);
    var x = pk;
    if (prm.movement == 1u) { x = d; }
    let n = CN[cidx(sj.x, st, sj.y)];
    var c = n;
    if (prm.color == 1u) { c = CS[cidx(sj.x, st, sj.y)]; }
    else if (prm.color == 2u) { c = length(d); }
    else if (prm.color == 3u) { c = 7.0; }
    reduceStats(x, n, c);
  }
  flushStats(li);
}

// Jeden wątek: dekodowanie granic, margines 4% (min. 0,03), środek i promień kamery.
// FRAME: 0 lo, 1 hi (z marginesem), 2 środek+promień, 3 surowe lo + max normy, 4 surowe hi + max koloru.
@compute @workgroup_size(1)
fn finalize_stats() {
  let lo = vec3f(unord(~atomicLoad(&STATS[0])), unord(~atomicLoad(&STATS[1])), unord(~atomicLoad(&STATS[2])));
  let hi = vec3f(unord(atomicLoad(&STATS[3])), unord(atomicLoad(&STATS[4])), unord(atomicLoad(&STATS[5])));
  let pad = max((hi - lo) * 0.04, vec3f(0.03));
  let plo = lo - pad;
  let phi = hi + pad;
  FRAME[0] = vec4f(plo, 0.0);
  FRAME[1] = vec4f(phi, 0.0);
  FRAME[2] = vec4f((plo + phi) * 0.5, length((phi - plo) * 0.5) * 1.07);
  FRAME[3] = vec4f(lo, unord(atomicLoad(&STATS[6])));
  FRAME[4] = vec4f(hi, max(unord(atomicLoad(&STATS[7])), 1e-6));
}

// ---------------------------------------------------------------------------
// Mapa ruchu 20 x 30: dla każdego patcha i stanu średnia normy 768D
// po 1500 obrazach (albo wartość jednego obrazu). Jedna grupa = (patch, stan).
var<workgroup> wsum: array<f32, 256>;

@compute @workgroup_size(256)
fn heat(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let t = wg.x % NTOK;
  let k = wg.x / NTOK;
  var acc = 0.0;
  if (prm.image == ALL) {
    for (var i = li; i < NIMG; i += 256u) { acc += nrm(i, k, t); }
  } else if (li == 0u) {
    acc = nrm(prm.image, k, t);
  }
  wsum[li] = acc;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (li < w) { wsum[li] += wsum[li + w]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    var n = 1.0;
    if (prm.image == ALL) { n = f32(NIMG); }
    HEAT[k * NTOK + t] = wsum[0] / n;
  }
}

// ---------------------------------------------------------------------------
// Dokładna mediana i maksimum normy bez sortowania: wybór pozycyjny po bitach.
// Normy są >= 0, więc porządek bitów u32 = porządek liczb.
// HIST[0..65535] starsze 16 bitów, HIST[65536] maksimum, HIST[65537..] młodsze 16 bitów.
fn normValue(e: u32) -> f32 {
  if (prm.source == 0u) {
    let it = particle(e);
    return nrm(it.x, prm.state, it.y);
  }
  let sj = slotOf(e);
  return CN[cidx(sj.x, prm.step, sj.y)];
}

@compute @workgroup_size(256)
fn hist_hi(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= prm.count) { return; }
  let bits = bitcast<u32>(normValue(g.x));
  atomicAdd(&HIST[bits >> 16u], 1u);
  atomicMax(&HIST[65536u], bits);
}

@compute @workgroup_size(256)
fn hist_lo(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= prm.count) { return; }
  let bits = bitcast<u32>(normValue(g.x));
  if ((bits >> 16u) == prm.prefix) { atomicAdd(&HIST[65537u + (bits & 0xffffu)], 1u); }
}

// ---------------------------------------------------------------------------
// Krzywe detekcji: dla każdego z 501 kroków średnia pewność, IoU i liczba
// obrazów spełniających oba progi 0,9. Jedna grupa = jeden krok.
var<workgroup> wc: array<f32, 256>;
var<workgroup> wq: array<f32, 256>;
var<workgroup> wn: array<u32, 256>;

@compute @workgroup_size(256)
fn curves(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let st = wg.x;
  var n = NIMG;
  if (prm.selKind == 1u) { n = 16u; } else if (prm.selKind == 2u) { n = 1u; }
  var sc = 0.0;
  var sq = 0.0;
  var cnt = 0u;
  for (var m = li; m < n; m += 256u) {
    var img = m;
    if (prm.selKind == 1u) { img = CIMG[m]; } else if (prm.selKind == 2u) { img = prm.selImage; }
    let c = CONF[img * H1 + st];
    let q = IOU[img * H1 + st];
    sc += c;
    sq += q;
    if (c >= 0.9 && q >= 0.9) { cnt += 1u; }
  }
  wc[li] = sc;
  wq[li] = sq;
  wn[li] = cnt;
  workgroupBarrier();
  for (var w = 128u; w > 0u; w >>= 1u) {
    if (li < w) {
      wc[li] += wc[li + w];
      wq[li] += wq[li + w];
      wn[li] += wn[li + w];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    let fn_ = f32(n);
    CURV[st * 4u] = wc[0] / fn_;
    CURV[st * 4u + 1u] = wq[0] / fn_;
    CURV[st * 4u + 2u] = f32(wn[0]) / fn_;
    CURV[st * 4u + 3u] = f32(wn[0]);
  }
}

// Średnia norma 768D każdego z 8 śledzonych patchy w panelu w danym kroku.
@compute @workgroup_size(8)
fn bars(@builtin(local_invocation_index) j: u32) {
  var acc = 0.0;
  var n = 1u;
  if (prm.image == ALL) {
    n = 16u;
    for (var sl = 0u; sl < 16u; sl++) { acc += CN[cidx(sl, prm.step, j)]; }
  } else {
    acc = CN[cidx(prm.image, prm.step, j)];
  }
  BARS[j] = acc / f32(n);
}
`;

// ---------------------------------------------------------------------------
// Renderowanie: shader wierzchołków czyta bufory policzone przez kernele.
export const RENDER = /* wgsl */`
struct Cam {
  yaw: f32, pitch: f32, zoom: f32, aspect: f32,
  size: f32, vw: f32, vh: f32, alpha: f32,
  colorMode: f32, lineLen: f32, lineAlpha: f32, lineIdentity: f32,
};
@group(0) @binding(0) var<uniform> cam: Cam;
@group(0) @binding(1) var<storage, read> PTS: array<vec4f>;
@group(0) @binding(2) var<storage, read> FR: array<vec4f>;
@group(0) @binding(3) var<storage, read> LN: array<vec4f>;

const PALETTE = array<vec3f, 8>(
  vec3f(0.384, 0.831, 0.867), vec3f(1.0, 0.725, 0.404), vec3f(0.847, 0.561, 0.898), vec3f(0.663, 0.843, 0.463),
  vec3f(0.553, 0.651, 1.0), vec3f(0.941, 0.565, 0.616), vec3f(0.875, 0.847, 0.537), vec3f(0.710, 0.808, 0.867));

fn project(p: vec3f) -> vec4f {
  let c = FR[2];
  var q = (p - c.xyz) / max(c.w, 1e-6);
  let cy = cos(cam.yaw);
  let sy = sin(cam.yaw);
  q = vec3f(cy * q.x + sy * q.z, q.y, -sy * q.x + cy * q.z);
  let cp = cos(cam.pitch);
  let sp = sin(cam.pitch);
  q = vec3f(q.x, cp * q.y - sp * q.z, sp * q.y + cp * q.z);
  return vec4f(q.x * cam.zoom / cam.aspect, q.y * cam.zoom, 0.5, 1.0);
}

fn ramp(t: f32) -> vec3f {
  let a = vec3f(0.325, 0.447, 0.545);
  let b = vec3f(0.384, 0.831, 0.867);
  let c = vec3f(1.0, 0.725, 0.404);
  if (t < 0.45) { return mix(a, b, t / 0.45); }
  return mix(b, c, (t - 0.45) / 0.55);
}

struct PointOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) rgb: vec3f,
  @location(2) @interpolate(flat) id: u32,
};

const CORNERS = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1), vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1));

@vertex
fn vs_points(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> PointOut {
  let d = PTS[i];
  var o: PointOut;
  let corner = CORNERS[v];
  var clip = project(d.xyz);
  clip = vec4f(clip.xy + corner * cam.size / vec2f(cam.vw, cam.vh), clip.zw);
  o.clip = clip;
  o.uv = corner;
  if (cam.colorMode > 2.5) {
    o.rgb = PALETTE[u32(d.w + 0.5) % 8u];
  } else {
    o.rgb = ramp(clamp(d.w / FR[4].w, 0.0, 1.0));
  }
  o.id = i + 1u;
  return o;
}

@fragment
fn fs_points(f: PointOut) -> @location(0) vec4f {
  if (dot(f.uv, f.uv) > 1.0) { discard; }
  return vec4f(f.rgb * cam.alpha, cam.alpha);
}

@fragment
fn fs_pick(f: PointOut) -> @location(0) u32 {
  if (dot(f.uv, f.uv) > 1.0) { discard; }
  return f.id;
}

struct LineOut {
  @builtin(position) clip: vec4f,
  @location(0) rgba: vec4f,
};

@vertex
fn vs_lines(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> LineOut {
  let n = u32(cam.lineLen);
  let h = v / 2u + v % 2u;
  let d = LN[i * n + h];
  var o: LineOut;
  o.clip = project(d.xyz);
  var rgb = vec3f(0.682, 0.741, 0.792);
  if (cam.lineIdentity > 0.5) { rgb = PALETTE[i % 8u]; }
  let a = cam.lineAlpha * (0.15 + 0.85 * d.w);
  o.rgba = vec4f(rgb * a, a);
  return o;
}

const EDGES = array<vec2u, 12>(
  vec2u(0, 1), vec2u(0, 2), vec2u(0, 4), vec2u(1, 3), vec2u(1, 5), vec2u(2, 3),
  vec2u(2, 6), vec2u(3, 7), vec2u(4, 5), vec2u(4, 6), vec2u(5, 7), vec2u(6, 7));

@vertex
fn vs_box(@builtin(vertex_index) v: u32) -> LineOut {
  let e = EDGES[v / 2u];
  var j = e.x;
  if (v % 2u == 1u) { j = e.y; }
  let lo = FR[0].xyz;
  let hi = FR[1].xyz;
  let p = vec3f(select(lo.x, hi.x, (j & 1u) != 0u), select(lo.y, hi.y, (j & 2u) != 0u), select(lo.z, hi.z, (j & 4u) != 0u));
  var o: LineOut;
  o.clip = project(p);
  o.rgba = vec4f(vec3f(0.337, 0.439, 0.525) * 0.45, 0.45);
  return o;
}

@fragment
fn fs_lines(f: LineOut) -> @location(0) vec4f { return f.rgba; }
`;
