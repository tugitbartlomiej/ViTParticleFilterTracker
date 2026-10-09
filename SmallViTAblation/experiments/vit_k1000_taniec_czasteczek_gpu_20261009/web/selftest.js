// Weryfikacja kerneli: GPU liczy, przeglądarka odczytuje wyniki, numpy na serwerze ocenia.
import { ALL, NIMG, NTOK, TRAIL_LEN, sha256 } from './gpu-engine.js';

async function postBlob(name, buffer) {
  const r = await fetch(`/api/verify/blob?name=${encodeURIComponent(name)}`, { method: 'POST', body: buffer });
  if (!r.ok) throw new Error(`Nie udało się wysłać ${name}`);
}

export async function runSelfTest(engine, meta, log) {
  engine.busy = true;
  const cases = [];
  const timings = {};
  try {
    // 1. Kinematyka całej chmury w zapisanych stanach: wynik ma być bitowo równy zapisowi.
    for (const movement of [0, 1]) {
      for (const k of [0, 1, 2]) {
        log(`Kinematyka 900 000 cząsteczek · stan ${[0, 100, 500][k]} · przesunięcia ${movement}`);
        await engine.setView({ kind: 'milestone', image: ALL, movement, color: 0 });
        engine.simulate({ ka: k, kb: k, s: 0 });
        const raw = await engine.readOut(NIMG * NTOK);
        const name = `kin_global_k${k}_m${movement}`;
        await postBlob(name, raw);
        cases.push({ name, kind: 'kin_milestone', blob: name, sha256: await sha256(raw),
          params: { image: -1, ka: k, kb: k, s: 0, movement, color: 0, bitwise: true } });
      }
    }
    // 2. Ruch między zapisami i kolory liczone na GPU (tolerancja: kolejność działań zmiennoprzecinkowych).
    const tol = [
      ['kin_global_odcinek_0_100_s037', { image: -1, ka: 0, kb: 1, s: 0.37, movement: 0, color: 0 }],
      ['kin_global_odcinek_100_500_s05_delta', { image: -1, ka: 1, kb: 2, s: 0.5, movement: 1, color: 2 }],
      ['kin_global_predkosc_100_500', { image: -1, ka: 1, kb: 2, s: 1, movement: 0, color: 1 }],
    ];
    for (const [name, p] of tol) {
      log(`Kinematyka · ${name}`);
      await engine.setView({ kind: 'milestone', image: ALL, movement: p.movement, color: p.color });
      engine.simulate(p);
      const raw = await engine.readOut(NIMG * NTOK);
      await postBlob(name, raw);
      cases.push({ name, kind: 'kin_milestone', blob: name, params: { ...p, bitwise: false } });
    }
    // 3. Jeden obraz.
    await engine.setView({ kind: 'milestone', image: 100, movement: 0, color: 0 });
    engine.simulate({ ka: 2, kb: 2, s: 0 });
    {
      const raw = await engine.readOut(NTOK);
      await postBlob('kin_obraz100_k2', raw);
      cases.push({ name: 'kin_obraz100_k2', kind: 'kin_milestone', blob: 'kin_obraz100_k2', sha256: await sha256(raw),
        params: { image: 100, ka: 2, kb: 2, s: 0, movement: 0, color: 0, bitwise: true } });
    }
    // 4. Panel kontrolny co krok i ślady.
    const ctrl = [
      ['kin_panel_wszystkie_krok100_delta', { panel: -1, step: 100, movement: 1, color: 0 }],
      ['kin_panel3_krok250_krok768', { panel: 3, step: 250, movement: 0, color: 1 }],
      ['kin_panel_wszystkie_krok500_deltaPCA', { panel: -1, step: 500, movement: 1, color: 2 }],
    ];
    for (const [name, p] of ctrl) {
      log(`Panel kontrolny · ${name}`);
      await engine.setView({ kind: 'control', image: p.panel < 0 ? ALL : p.panel, movement: p.movement, color: p.color });
      engine.simulate({ step: p.step });
      const raw = await engine.readOut(engine.view.count);
      await postBlob(name, raw);
      cases.push({ name, kind: 'kin_control', blob: name, params: p });
    }
    for (const [name, p] of [['slad_wszystkie_krok100_delta', { panel: -1, step: 100, movement: 1 }], ['slad_panel5_krok20', { panel: 5, step: 20, movement: 0 }]]) {
      await engine.setView({ kind: 'control', image: p.panel < 0 ? ALL : p.panel, movement: p.movement, color: 3, trails: true });
      engine.simulate({ step: p.step });
      const raw = await engine.readTrail(engine.view.count * (TRAIL_LEN + 1));
      await postBlob(name, raw);
      cases.push({ name, kind: 'trail', blob: name, params: p });
    }
    // 5. Granice osi i maksima (redukcje atomowe).
    log('Redukcje granic osi');
    for (const [name, image, movement] of [['granice_cala_chmura', ALL, 0], ['granice_cala_chmura_delta', ALL, 1], ['granice_obraz100_delta', 100, 1]]) {
      const f = await engine.setView({ kind: 'milestone', image, movement, color: 0 });
      cases.push({ name, kind: 'stats_milestone', params: { image: image === ALL ? -1 : image, movement }, result: { lo: f.rawLo, hi: f.rawHi, norm_max: f.normMax } });
    }
    for (const [name, panel, movement] of [['granice_panel', ALL, 0], ['granice_panel_delta', ALL, 1], ['granice_panel7', 7, 0]]) {
      const f = await engine.setView({ kind: 'control', image: panel, movement, color: 3 });
      cases.push({ name, kind: 'stats_control', params: { panel: panel === ALL ? -1 : panel, movement }, result: { lo: f.rawLo, hi: f.rawHi, norm_max: f.normMax } });
    }
    // 6. Mapy ruchu, mediany, krzywe, słupki.
    log('Mapy, mediany i krzywe detekcji');
    cases.push({ name: 'mapa_1500_obrazow', kind: 'heat', params: { image: -1 }, result: Array.from(await engine.heat(ALL)) });
    cases.push({ name: 'mapa_obraz42', kind: 'heat', params: { image: 42 }, result: Array.from(await engine.heat(42)) });
    for (const state of [0, 1, 2]) {
      cases.push({ name: `mediana_900000_stan${state}`, kind: 'order_stats', params: { source: 'milestone', image: -1, state },
        result: await engine.orderStats({ source: 0, image: ALL, state }) });
    }
    cases.push({ name: 'mediana_obraz100_stan2', kind: 'order_stats', params: { source: 'milestone', image: 100, state: 2 },
      result: await engine.orderStats({ source: 0, image: 100, state: 2 }) });
    cases.push({ name: 'mediana_panel_krok250', kind: 'order_stats', params: { source: 'control', panel: -1, step: 250 },
      result: await engine.orderStats({ source: 1, image: ALL, step: 250 }) });
    for (const [name, k, image] of [['krzywe_1500', 0, 0], ['krzywe_panel16', 1, 0], ['krzywe_obraz400', 2, 400]]) {
      cases.push({ name, kind: 'curves', params: { kind: k, image }, result: Array.from(await engine.curves(k, image)) });
    }
    cases.push({ name: 'slupki_panel_krok100', kind: 'bars', params: { panel: -1, step: 100 }, result: Array.from(await engine.bars(ALL, 100)) });
    cases.push({ name: 'slupki_panel2_krok300', kind: 'bars', params: { panel: 2, step: 300 }, result: Array.from(await engine.bars(2, 300)) });

    // 7. Wybór punktu: ostatnia rysowana cząsteczka leży na wierzchu, więc kliknięcie w jej środek musi ją zwrócić.
    log('Wybór punktu przez teksturę identyfikatorów');
    await engine.setView({ kind: 'milestone', image: 100, movement: 0, color: 0 });
    engine.simulate({ ka: 2, kb: 2, s: 0 });
    engine.draw();
    const outRaw = new Float32Array(await engine.readOut(NTOK));
    const last = NTOK - 1;
    const [sx, sy] = engine.projectToScreen([outRaw[last * 4], outRaw[last * 4 + 1], outRaw[last * 4 + 2]]);
    const hit = await engine.pick(sx, sy);
    cases.push({ name: 'wybor_punktu_obraz100_patch599', kind: 'pick', params: { expected_id: last, x: sx, y: sy }, result: { id: hit ? hit.id : -1 } });

    // 8. Czas kernela ruchu dla 900 000 cząsteczek (ściana zegara, z narzutem kolejki).
    log('Pomiar czasu kernela ruchu');
    await engine.setView({ kind: 'milestone', image: ALL, movement: 1, color: 0 });
    await engine.idle();
    const reps = 200, t0 = performance.now();
    for (let i = 0; i < reps; i++) engine.simulate({ ka: 0, kb: 1, s: (i % 100) / 100 });
    await engine.idle();
    timings.kin_milestone_900000_ms_per_dispatch_wall = (performance.now() - t0) / reps;
    timings.scope = 'Średni czas ściany na wywołanie (zapis parametrów + dispatch + submit), 200 powtórzeń, bez rysowania. To nie jest izolowany pomiar GPU.';
    timings.uploaded_bytes = engine.uploadedBytes;

    log('Ocena przez numpy na serwerze…');
    const r = await fetch('/api/verify/finish', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cases, timings, gpu: { adapter: engine.adapterInfo, userAgent: navigator.userAgent } }) });
    if (!r.ok) throw new Error(`Sędzia numpy zwrócił błąd ${r.status}`);
    return await r.json();
  } finally {
    engine.busy = false;
  }
}
