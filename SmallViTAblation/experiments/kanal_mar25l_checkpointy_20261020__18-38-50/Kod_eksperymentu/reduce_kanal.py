"""Redukcja (CPU) wyników testu kanału Mar25l: tabele CSV, decyzje hipotez, bootstrap, przedziały czasowe.

Wejście: Wyniki/Dane/seeds/s<seed>/k<step>/ (zweryfikowane SHA z SEED_COMPLETE.json).
Wyjście: Wyniki/Dane/redukcja/ (tylko dane; wykresy, raport i wniosek przygotowuje się osobno po zgodzie użytkownika).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kanal_io import file_sha, read_json, save_json, verify_freeze  # noqa: E402

W_KEYS = ('c_out', 'c_in_eff', 'c_in_raw', 'd_in_eff_rel', 'd_in_raw_rel', 'm_norm', 's_norm', 'sym_to_anti',
          'a_balance', 'm_times_d_eff', 'm_times_d_raw', 'pair_param_norm')
U_KEYS = ('f_channel', 'proj_m', 'proj_delta', 'share_vs_uniform')
T_KEYS = ('rho', 'kappa', 'glu_error', 'taylor_error', 'coactivation', 'r_z', 'delta_rel', 'y_share', 'u_share',
          'anti_share', 'rho_coactive')


def token_metrics(t, names):
    c = {n: t[:, i] for i, n in enumerate(names)}
    n = c['n']
    with np.errstate(invalid='ignore', divide='ignore'):
        mi, mj = c['zi'] / n, c['zj'] / n
        vi, vj = c['zi2'] / n - mi ** 2, c['zj2'] / n - mj ** 2
        return dict(rho=c['y_norm'] / c['u_sum'], kappa=c['y2'] / c['u2'], glu_error=np.sqrt(c['err2'] / c['y2']),
                    taylor_error=np.sqrt(c['taylor2'] / c['dg2']), coactivation=c['coact'] / n,
                    r_z=(c['zizj'] / n - mi * mj) / np.sqrt(vi * vj), delta_rel=np.sqrt(c['delta2'] / c['zsq']),
                    y_share=c['y_share'] / n, u_share=c['u_share'] / n, anti_share=c['anti2'] / (c['anti2'] + c['sym2']),
                    rho_coactive=c['y_norm_coact'] / c['u_sum_coact'])


def med(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float('nan')


def boot_seeds(values, reps, rng):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) < 2:
        return dict(n=len(v), median=med(v), ci_low=None, ci_high=None)
    b = np.median(v[rng.integers(0, len(v), (reps, len(v)))], axis=1)
    return dict(n=len(v), median=float(np.median(v)), mean=float(v.mean()), ci_low=float(np.quantile(b, .025)),
                ci_high=float(np.quantile(b, .975)))


def boot_crossed(M, reps, rng):
    """M[seed, film]: średnia po komórkach przy losowaniu seedów i filmów ze zwracaniem (brak pseudoreplikacji)."""
    M = np.asarray(M, dtype=np.float64)
    ok = np.isfinite(M)
    if ok.sum() == 0:
        return dict(mean=None, ci_low=None, ci_high=None)
    S, F = M.shape
    out = np.empty(reps)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # puste losowania dają NaN, pomijane w kwantylach
        for r in range(reps):
            s = rng.integers(0, S, S); f = rng.integers(0, F, F)
            out[r] = np.nanmean(M[np.ix_(s, f)])
    return dict(mean=float(np.nanmean(M)), ci_low=float(np.nanquantile(out, .025)), ci_high=float(np.nanquantile(out, .975)),
                seeds=int(S), films=int(F), finite_cells=int(ok.sum()))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if (isinstance(v, float) and not math.isfinite(v)) else v) for k, v in r.items()})


def interval_of_max(steps, values, sign=+1):
    v = np.asarray(values, dtype=np.float64)
    d = sign * np.diff(v)
    if not np.isfinite(d).any():
        return None
    i = int(np.nanargmax(d))
    return f'{steps[i]}-{steps[i + 1]}'


def reduce(exp, seeds_dir, out, reps, smoke=False):
    """smoke=True oznacza bieg częściowy (smoke lub pilot): dowolna liczba seedów, kroki z SEED_COMPLETE."""
    cfg = read_json(HERE / 'config.json')
    H = cfg['hypotheses']
    rng = np.random.default_rng(cfg['bootstrap']['seed'])
    seed_dirs = sorted([p for p in seeds_dir.glob('s*') if (p / 'SEED_COMPLETE.json').exists()], key=lambda p: int(p.name[1:]))
    if not seed_dirs:
        raise ValueError('No complete seeds in ' + str(seeds_dir))
    steps = read_json(seed_dirs[0] / 'SEED_COMPLETE.json')['steps']
    if any(read_json(s / 'SEED_COMPLETE.json')['steps'] != steps for s in seed_dirs):
        raise ValueError('Seeds differ in measured steps')
    if not smoke and (len(seed_dirs) != len(cfg['seeds']) or steps != cfg['steps']):
        raise ValueError(f'Expected {len(cfg["seeds"])} complete seeds with all steps, found {len(seed_dirs)}')
    verified = {}
    for sd in seed_dirs:
        done = read_json(sd / 'SEED_COMPLETE.json')
        for name, sha in done['files'].items():
            if file_sha(sd / name) != sha:
                raise ValueError('Changed result ' + str(sd / name))
        verified[sd.name] = file_sha(sd / 'SEED_COMPLETE.json')
    rows_w, rows_t, rows_h, rows_pop, rows_popd, rows_det = [], [], [], [], [], []
    per_seed = defaultdict(dict)  # (seed) -> dict of primary values
    film_cells = defaultdict(lambda: defaultdict(dict))  # metric -> panel -> seed -> film vector
    timing_series = defaultdict(lambda: defaultdict(list))
    literal_total = defaultdict(int)
    for sd in seed_dirs:
        seed = int(sd.name[1:])
        for step in steps:
            kd = sd / f'k{step:05d}'
            summ = read_json(kd / 'STEP_SUMMARY.json')
            for panel, pdet in summ['panels'].items():
                rows_det.append(dict(seed=seed, step=step, panel=panel, **pdet))
                timing_series[(seed, 'iou_' + panel)]['v'].append(pdet['iou_mean_gt'] if pdet['iou_mean_gt'] is not None else np.nan)
            for b in summ['blocks']:
                k = b['block']
                pr = dict(seed=seed, step=step, block=k, opposed_pairs=b['opposed_pairs'], with_opponent=b['with_opponent'],
                          spearman_cin_eff_cout=b['spearman_cin_eff_cout'], spearman_cin_raw_cout=b['spearman_cin_raw_cout'],
                          max_pair_param_norm=b['max_pair_param_norm'], a_norm_median=b['a_norm_median'])
                for key, e in b['enrichment'].items():
                    pr[f'enrich_{key}'] = e['ratio']; pr[f'cond_{key}'] = e['conditional']; pr[f'base_{key}'] = e['base']
                pr.update({f'q_{kk}': v for kk, v in b['quadrants'].items()})
                pr.update({f'litpop_{kk}': v for kk, v in b['literal_population'].items()})
                pr.update({f'littrack_{kk}': v for kk, v in b['literal_tracked'].items()})
                for sname, info in b['sets'].items():
                    pr[f'set_{sname}_total'] = info['total']
                rows_pop.append(pr)
                literal_total[k] += b['literal_tracked']['all']
                for panel, dd in b.get('data', {}).items():
                    rows_popd.append(dict(seed=seed, step=step, block=k, panel=panel, tokens=dd['tokens'],
                                          kappa_median_all=dd['all']['kappa_median'], c_g_median_all=dd['all']['c_g_median'],
                                          **{f'opposed_{kk}': v for kk, v in dd['opposed'].items()}))
                p = np.load(kd / f'pairs_b{k}.npz', allow_pickle=False)
                mem = p['membership']; set_names = [str(s) for s in p['set_names']]; tnames = [str(s) for s in p['token_sum_names']]
                tm = {panel: token_metrics(p['t_' + panel], tnames) for panel in [kk[2:] for kk in p.files if kk.startswith('t_')]}
                rnd_mask = ((mem >> set_names.index('random')) & 1) == 1
                rnd_q95 = float(np.quantile(p['w_c_in_eff'][rnd_mask], H['H0_no_channel']['median_c_in_eff_opposed_final_le_random_quantile'])) if rnd_mask.any() else float('nan')
                for bit, sname in enumerate(set_names):
                    m = ((mem >> bit) & 1) == 1
                    rw = dict(seed=seed, step=step, block=k, set=sname, n=int(m.sum()), total=b['sets'][sname]['total'],
                              truncated=b['sets'][sname]['truncated'])
                    for key in W_KEYS:
                        rw[key] = med(p['w_' + key][m]) if m.any() else float('nan')
                    for key in U_KEYS:
                        rw[key] = med(p['u_' + key][m]) if ('u_' + key in p.files and m.any()) else float('nan')
                    if 'u_f_channel' in p.files and m.any():
                        rw['f_channel_mean'] = float(np.nanmean(p['u_f_channel'][m]))
                        rw['f_channel_random_expectation'] = float(p['u_random_expectation'][0])
                    rw['n_literal_all'] = int(p['l_all'][m].sum()) if m.any() else 0
                    rows_w.append(rw)
                    for panel, mt in tm.items():
                        rt = dict(seed=seed, step=step, block=k, set=sname, panel=panel, n=int(m.sum()))
                        for key in T_KEYS:
                            rt[key] = med(mt[key][m]) if m.any() else float('nan')
                        rows_t.append(rt)
                    if k == H['primary_block'] and step == H['primary_step']:
                        ps = per_seed[seed]
                        ps[f'{sname}_n'] = int(m.sum())
                        ps[f'{sname}_c_in_eff'] = rw['c_in_eff']
                        for panel, mt in tm.items():
                            ps[f'{sname}_kappa_{panel}'] = med(mt['kappa'][m]) if m.any() else float('nan')
                            ps[f'{sname}_glu_{panel}'] = med(mt['glu_error'][m]) if m.any() else float('nan')
                            ps[f'{sname}_rho_{panel}'] = med(mt['rho'][m]) if m.any() else float('nan')
                    if k == H['primary_block'] and sname in ('opposed_now', 'opposed_final', 'near_channel_now'):
                        timing_series[(seed, f'{sname}_c_in_eff')]['v'].append(rw['c_in_eff'])
                        timing_series[(seed, f'{sname}_c_out')]['v'].append(rw['c_out'])
                        timing_series[(seed, f'{sname}_n')]['v'].append(b['sets'][sname]['total'])
                        if 'test' in tm:
                            timing_series[(seed, f'{sname}_kappa_test')]['v'].append(med(tm['test']['kappa'][m]) if m.any() else np.nan)
                if k == H['primary_block'] and step == H['primary_step']:
                    ps = per_seed[seed]
                    ps['random_c_in_eff_q95'] = rnd_q95
                    ps['enrichment_eff_0.9'] = b['enrichment']['eff_0.9']['ratio']
                    ps['opposed_now_total'] = b['sets']['opposed_now']['total']
                    ps['near_channel_total'] = b['sets']['near_channel_now']['total']
                    # komórki seed×film dla bootstrapu krzyżowego
                    fnames = [str(s) for s in p['f_sum_names']]; fpos = p['f_positions']
                    for panel in tm:
                        if 'f_' + panel not in p.files:
                            continue
                        F = p['f_' + panel]
                        c = {n: F[:, :, i] for i, n in enumerate(fnames)}
                        with np.errstate(invalid='ignore', divide='ignore'):
                            fm = dict(kappa=c['y2'] / c['u2'], glu_error=np.sqrt(c['err2'] / c['y2']), rho=c['y_norm'] / c['u_sum'],
                                      coactivation=c['coact'] / c['n'])
                        memf = mem[fpos]
                        for sname in ('opposed_final', 'functional', 'random', 'near_channel_now'):
                            ms = ((memf >> set_names.index(sname)) & 1) == 1
                            for metric, val in fm.items():
                                cells = np.array([med(val[ms, f]) if ms.any() else np.nan for f in range(F.shape[1])])
                                film_cells[f'{sname}_{metric}'][panel][seed] = cells
                if k == H['primary_block']:
                    if 'h_test' in p.files or 'h_val' in p.files:
                        hn = [str(s) for s in p['h_sum_names']]; an = [str(s) for s in p['h_anchor_names']]; hpos = p['h_positions']
                        memh = mem[hpos]
                        for panel in tm:
                            h = p['h_' + panel]
                            for ai, anchor in enumerate(an):
                                c = {n: h[:, ai, i] for i, n in enumerate(hn)}
                                with np.errstate(invalid='ignore', divide='ignore'):
                                    fid = np.sqrt(c['glu_edge2'] / c['abl_edge2'])
                                    abl_rms = np.sqrt(c['abl_edge2'] / (4 * c['n']))
                                    abl_logit = c['abl_logit'] / c['n']
                                for sname in ('functional', 'near_channel_now', 'literal_now', 'opposed_final', 'random'):
                                    ms = ((memh >> set_names.index(sname)) & 1) == 1
                                    rows_h.append(dict(seed=seed, step=step, panel=panel, anchor=anchor, set=sname, n=int(ms.sum()),
                                                       glu_fidelity_ratio=med(fid[ms]) if ms.any() else float('nan'),
                                                       ablation_rms_edge_px=med(abl_rms[ms]) if ms.any() else float('nan'),
                                                       ablation_logit_mean=med(abl_logit[ms]) if ms.any() else float('nan')))
                p.close()
    red = out
    write_csv(red / 'SETS_WEIGHTS.csv', rows_w)
    write_csv(red / 'SETS_TOKENS.csv', rows_t)
    write_csv(red / 'HEAD_GLU.csv', rows_h)
    write_csv(red / 'POPULATION.csv', rows_pop)
    write_csv(red / 'POPULATION_DATA.csv', rows_popd)
    write_csv(red / 'DETECTION.csv', rows_det)
    # ---- hipotezy (blok i krok główny), panel główny i replikacja
    h1, h0, h1s = H['H1_channel'], H['H0_no_channel'], H['H1s_channel_subpopulation']
    decisions = {}
    for panel in (H['primary_panel'], H['replication_panel']):
        rec = {}
        for seed, ps in sorted(per_seed.items()):
            if f'opposed_final_kappa_{panel}' not in ps:
                continue
            c = dict(a=ps['opposed_final_c_in_eff'] >= h1['median_c_in_eff_opposed_final_ge'],
                     b=(ps['enrichment_eff_0.9'] or 0) >= h1['enrichment_ratio_tau_0_9_ge'],
                     c=ps[f'opposed_final_kappa_{panel}'] <= h1['median_kappa_opposed_final_le'],
                     d=ps[f'opposed_final_glu_{panel}'] <= h1['median_glu_error_opposed_final_le'])
            frac_near = ps['near_channel_total'] / ps['opposed_now_total'] if ps['opposed_now_total'] else float('nan')
            z0 = dict(median_at_or_below_random_q95=ps['opposed_final_c_in_eff'] <= ps['random_c_in_eff_q95'],
                      near_channel_fraction_small=(frac_near < h0['near_channel_fraction_of_opposed_lt']) if np.isfinite(frac_near) else True)
            s1 = dict(nonempty=ps['near_channel_total'] > 0,
                      kappa_low=(ps.get(f'near_channel_now_kappa_{panel}', float('nan')) <= h1s['median_kappa_near_channel_le']))
            rec[seed] = dict(values=ps, near_channel_fraction=frac_near, H1=c, H1_all=all(c.values()), H0=z0, H0_all=all(z0.values()),
                             H1s=s1, H1s_all=all(s1.values()))
        n = len(rec)
        cnt = dict(H1=sum(r['H1_all'] for r in rec.values()), H0=sum(r['H0_all'] for r in rec.values()),
                   H1s=sum(r['H1s_all'] for r in rec.values()))
        need = H['seeds_required'] if not smoke else max(1, n)
        verdict = ('H1_channel' if cnt['H1'] >= need else 'H0_no_channel' if cnt['H0'] >= need else 'mixed')
        decisions[panel] = dict(seeds=n, counts=cnt, seeds_required=need, verdict=verdict,
                                subpopulation_supported=cnt['H1s'] >= need, per_seed=rec)
    # ---- bootstrap po seedach i krzyżowy seed×film
    keys = ['opposed_final_c_in_eff', 'random_c_in_eff_q95', 'enrichment_eff_0.9', 'near_channel_total', 'opposed_now_total']
    for panel in (H['primary_panel'], H['replication_panel']):
        keys += [f'{s}_{m}_{panel}' for s in ('opposed_final', 'functional', 'random', 'twins_now') for m in ('kappa', 'glu', 'rho')]
    seed_boot = {k: boot_seeds([ps.get(k) for ps in per_seed.values()], reps, rng) for k in keys}
    crossed = {}
    for metric, panels in film_cells.items():
        for panel, by_seed in panels.items():
            M = np.stack([by_seed[s] for s in sorted(by_seed)])
            crossed[f'{metric}_{panel}'] = boot_crossed(M, reps, rng)
    # ---- przedziały największych zmian (blok główny)
    timing = defaultdict(lambda: defaultdict(int))
    rules = {'opposed_now_n': +1, 'opposed_final_c_in_eff': +1, 'opposed_final_c_out': -1, 'near_channel_now_n': +1,
             'opposed_final_kappa_test': -1, 'iou_test': +1, 'iou_val': +1}
    timing_per_seed = defaultdict(dict)
    for (seed, name), d in timing_series.items():
        if name in rules:
            iv = interval_of_max(steps, d['v'], rules[name])
            timing_per_seed[name][seed] = iv
            timing[name][str(iv)] += 1
    result = dict(schema=1, experiment=cfg['experiment'], smoke=smoke, seeds=[int(s.name[1:]) for s in seed_dirs], steps=steps,
                  verified_seed_complete_sha256=verified, hypotheses_config=H, decisions=decisions,
                  seed_bootstrap=seed_boot, crossed_bootstrap=crossed,
                  timing_counts={k: dict(v) for k, v in timing.items()}, timing_per_seed={k: dict(v) for k, v in timing_per_seed.items()},
                  timing_rules={k: ('largest increase' if v > 0 else 'largest decrease') for k, v in rules.items()},
                  literal_all_tracked_by_block=dict(literal_total), bootstrap_replicates=reps)
    save_json(red / 'HYPOTHESES_AND_SUMMARY.json', result)
    files = {p.name: file_sha(p) for p in sorted(red.iterdir()) if p.is_file() and p.name != 'REDUCE_MANIFEST.json'}
    save_json(red / 'REDUCE_MANIFEST.json', dict(files=files, inputs=verified))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp-root'); p.add_argument('--seeds-dir'); p.add_argument('--out'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--pilot', action='store_true'); p.add_argument('--reps', type=int); p.add_argument('--no-freeze', action='store_true')
    a = p.parse_args()
    exp = Path(a.exp_root) if a.exp_root else HERE.parent
    if not a.no_freeze:
        verify_freeze(exp)
    cfg = read_json(HERE / 'config.json')
    kind = 'smoke' if a.smoke else 'pilot' if a.pilot else 'seeds'
    seeds_dir = Path(a.seeds_dir) if a.seeds_dir else exp / 'Wyniki' / 'Dane' / kind
    out = Path(a.out) if a.out else exp / 'Wyniki' / 'Dane' / ('redukcja' if kind == 'seeds' else f'redukcja_{kind}')
    r = reduce(exp, seeds_dir, out, a.reps or cfg['bootstrap']['replicates'], smoke=kind != 'seeds')
    print({p: d['verdict'] for p, d in r['decisions'].items()}, flush=True)


if __name__ == '__main__':
    main()
