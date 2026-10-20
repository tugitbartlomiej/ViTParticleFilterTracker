"""Worker testu kanału Mar25l: jeden seed, 12 checkpointów, 6 bloków MLP, panele TEST50 i VAL50.

Tylko inferencja. Źródła (wagi, obrazy, katalogi par) są czytane i weryfikowane SHA256, nigdy zmieniane.
Wyniki: <exp>/Wyniki/Dane/seeds/s<seed>/k<step>/ (produkcja) lub <exp>/Wyniki/Dane/smoke/ (smoke).
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kanal_core as K  # noqa: E402
from kanal_io import (file_sha, read_json, save_json, save_npz, verify_npz, seed_lock, verify_freeze,  # noqa: E402
                      done_path)

SET_NAMES = ('opposed_now', 'opposed_final', 'near_channel_now', 'copies_now', 'twins_now', 'literal_now',
             'functional', 'random')
FILM_SETS = ('opposed_final', 'functional', 'near_channel_now', 'random')
FILM_SUMS = ('n', 'y_norm', 'u_sum', 'y2', 'u2', 'err2', 'taylor2', 'dg2', 'coact')
HEAD_PRIORITY = ('functional', 'near_channel_now', 'literal_now', 'opposed_final', 'random')
HEAD_SUMS = ('n', 'abl_edge2', 'glu_edge2', 'abl_logit', 'abl_logit2', 'glu_logit', 'glu_logit2')
ANCHORS = ('clean_winner', 'gt_anchor')
WEIGHT_KEYS = None  # ustalane przy pierwszym wywołaniu pair_weight_metrics
TOL = dict(fc1=5e-4, gelu=1e-5, fc2=5e-4, post=1e-4, head=2e-4, hook=2e-4, vector=1e-8)


# ----------------------------------------------------------------------------------------------- wejścia
def resolve_paths(cfg, args):
    exp = Path(args.exp_root) if args.exp_root else HERE.parent
    source = Path(args.source_root or cfg['source_root'])
    pruning = Path(args.pruning_root or cfg['pruning_root'])
    return exp, source, pruning


def prepare_inputs(cfg, source, pruning, seed):
    src_freeze = read_json(source / 'FREEZE.json')['files']
    prov = dict(source_root=str(source), pruning_root=str(pruning), source_files={}, seed=seed)
    for name in ('code/model.py', 'code/core.py', 'input/dataset.json'):
        sha = file_sha(source / name)
        if sha != src_freeze[name]:
            raise ValueError('Source file changed: ' + name)
        prov['source_files'][name] = sha
    prov['source_freeze_sha256'] = file_sha(source / 'FREEZE.json')
    # SHA wag: SOURCE_PROVENANCE.json kampanii pruning (zweryfikowany jej FREEZE i SHA źródła); na Edenie źródło nie ma
    # WEIGHTS_MANIFEST.json (powstał przy eksporcie) — jeśli istnieje (kopia lokalna), musi się zgadzać.
    pr_freeze = read_json(pruning / 'FREEZE.json')['files']
    if file_sha(pruning / 'SOURCE_PROVENANCE.json') != pr_freeze.get('SOURCE_PROVENANCE.json'):
        raise ValueError('Pruning SOURCE_PROVENANCE changed')
    src_prov = read_json(pruning / 'SOURCE_PROVENANCE.json')
    if src_prov['freeze_sha256'] != prov['source_freeze_sha256']:
        raise ValueError('Pruning provenance refers to a different source freeze')
    weights = {k: v for k, v in src_prov['files'].items() if k.startswith('replications/')}
    prov['weights_sha_source'] = 'pruning SOURCE_PROVENANCE.json'
    if (source / 'WEIGHTS_MANIFEST.json').exists():
        wm = read_json(source / 'WEIGHTS_MANIFEST.json')['files']
        if any(wm.get(k) != v for k, v in weights.items()):
            raise ValueError('WEIGHTS_MANIFEST disagrees with pruning provenance')
        prov['weights_sha_source'] += ' + WEIGHTS_MANIFEST.json (agree)'
    seed_done = read_json(source / f'results/s{seed}/SEED_COMPLETE.json')['files']
    geo_name = f'results/s{seed}/geometry/GEOMETRY.json'
    if file_sha(source / geo_name) != seed_done[geo_name]:
        raise ValueError('Source geometry changed')
    geometry = read_json(source / geo_name)
    prov['geometry_sha256'] = seed_done[geo_name]
    pr_done = read_json(pruning / f'results/s{seed}/SEED_COMPLETE.json')
    cat_name = f'results/s{seed}/CATALOGS.json'
    if pr_done.get('status') != 'complete' or file_sha(pruning / cat_name) != pr_done['files'][cat_name]:
        raise ValueError('Pruning catalog changed or incomplete')
    catalog = read_json(pruning / cat_name)
    prov['catalog_sha256'] = pr_done['files'][cat_name]
    prov['pruning_freeze_sha256'] = pr_done['freeze_sha256']
    functional = [tuple(sorted(map(int, p))) for p in catalog['final_families']['functional']['pairs']]
    rows = read_json(source / 'input/dataset.json')['rows']
    panels = build_panels(rows, cfg)
    return dict(weights=weights, geometry=geometry, functional=functional, panels=panels, prov=prov)


def build_panels(rows, cfg):
    panels = {}
    for name, spec in cfg['panels'].items():
        films = defaultdict(list)
        for r in rows:
            if r['role'] == spec['role']:
                films[r['film']].append(r)
        if len(films) != spec['films'] or any(len(v) != spec['frames_per_film'] for v in films.values()):
            raise ValueError('Panel source changed: ' + name)
        chosen = []
        for fi, film in enumerate(sorted(films)):
            ordered = sorted(films[film], key=lambda r: (r['frame'], r['id']))
            for pos in spec['positions']:
                chosen.append(dict(ordered[pos], film_index=fi))
        panels[name] = chosen
    return panels


def load_image(source, row):
    path = source / 'input' / row['image']
    if file_sha(path) != row['image_sha256']:
        raise ValueError('Image changed: ' + row['id'])
    with np.load(path, allow_pickle=False) as data:
        return data['rgb']


def checkpoint_path(source, seed, step):
    return source / f'replications/s{seed}/weights/k{step:05d}.pt'


# ----------------------------------------------------------------------------------------------- model
class Taps:
    """Haki: wejście MLP (residual), wyjście norm2, preaktywacja fc1, GELU, wyjście fc2, wejście końcowej normy."""

    def __init__(self, torch, net):
        self.d = {}
        for k, b in enumerate(net.bloki):
            if not (isinstance(b.mlp.act, torch.nn.GELU) and b.mlp.act.approximate == 'none'):
                raise ValueError('Unexpected activation')
            for name in ('ls2', 'drop_path2'):
                if not isinstance(getattr(b, name), torch.nn.Identity):
                    raise ValueError('Unexpected residual branch module ' + name)
            if not isinstance(b.mlp.norm, torch.nn.Identity):
                raise ValueError('Unexpected MLP norm')
            b.norm2.register_forward_pre_hook(lambda m, a, k=k: self.d.__setitem__(('res', k), a[0].detach()))
            b.norm2.register_forward_hook(lambda m, a, o, k=k: self.d.__setitem__(('xin', k), o.detach()))
            b.mlp.fc1.register_forward_hook(lambda m, a, o, k=k: self.d.__setitem__(('z', k), o.detach()))
            b.mlp.act.register_forward_hook(lambda m, a, o, k=k: self.d.__setitem__(('g', k), o.detach()))
            b.mlp.fc2.register_forward_hook(lambda m, a, o, k=k: self.d.__setitem__(('out', k), o.detach()))
        net.norm.register_forward_pre_hook(lambda m, a: self.d.__setitem__('post', a[0].detach()))


def block_weights(torch, net, device):
    out = []
    for b in net.bloki:
        W1 = b.mlp.fc1.weight.detach().double().to(device); b1 = b.mlp.fc1.bias.detach().double().to(device)
        gam = b.norm2.weight.detach().double().to(device); bet = b.norm2.bias.detach().double().to(device)
        V = b.mlp.fc2.weight.detach().double().to(device).T.contiguous(); b2 = b.mlp.fc2.bias.detach().double().to(device)
        th_eff, th_raw, b_hat = K.effective_input(W1, b1, gam, bet)
        out.append(dict(W1=W1, b1=b1, V=V, b2=b2, theta_eff=th_eff, theta_raw=th_raw, b_hat=b_hat, A=V @ V.T))
    return out


def geometry_check(state, expected):
    """Dokładnie ten sam rachunek co źródłowe GEOMETRY.json (numpy float64, CPU)."""
    for k in range(6):
        vector = state[f'bloki.{k}.mlp.fc2.weight'].double().numpy().T
        norm = np.linalg.norm(vector, axis=1)
        vn = vector / np.maximum(norm[:, None], 1e-30)
        cs = vn @ vn.T
        np.fill_diagonal(cs, 1)
        exp = expected[k]
        if int((cs < -.5).any(1).sum()) != exp['with_opponent'] or int(np.triu(cs < -.5, 1).sum()) != exp['opposed_pairs']:
            raise ValueError(f'Geometry count differs from source, block {k}')
        if not (np.isclose(cs.min(1).mean(), exp['mean_min_cos'], rtol=1e-9, atol=1e-12)
                and np.isclose(norm.mean(), exp['mean_norm'], rtol=1e-9, atol=1e-12)):
            raise ValueError(f'Geometry scalar differs from source, block {k}')


def decode(torch, raw, tok):
    """raw (...,5) dla tokenu tok → (logit, x0, y0, x1, y1) w pikselach (siatka 20×30, krok 16)."""
    d = torch.exp(raw[..., 1:]) * 16
    cx = (tok % 30 + .5) * 16; cy = (tok // 30 + .5) * 16
    return torch.stack([raw[..., 0], cx - d[..., 0], cy - d[..., 1], cx + d[..., 2], cy + d[..., 3]], -1)


def anchors_for(raw0, gt):
    out = {'clean_winner': int(np.argmax(raw0))}
    if len(gt):
        g = max(gt, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cx = (g[0] + g[2]) / 2; cy = (g[1] + g[3]) / 2
        out['gt_anchor'] = int(np.clip(round(cy / 16 - .5), 0, 19)) * 30 + int(np.clip(round(cx / 16 - .5), 0, 29))
    return out


# ----------------------------------------------------------------------------------------------- zbiory par
def pid_of(i, j, n):
    i, j = min(i, j), max(i, j)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def build_sets(torch, cu, cfg, block, last_block, final_sets, functional_pids, random_pids):
    s = cfg['sets']
    info = {}
    sets = {}

    def put(name, sel):
        idx, total, trunc = sel
        sets[name] = idx
        info[name] = dict(selected=int(idx.numel()), total=int(total), truncated=bool(trunc))

    put('opposed_now', K.select_top(cu['c_out'], cu['c_out'] < s['opposed_cos_lt'], s['cap'], largest=False))
    sets['opposed_final'] = final_sets[block]['idx']
    info['opposed_final'] = dict(final_sets[block]['info'])
    near = (cu['c_in_eff'] > s['near_channel_cin_gt']) & (cu['c_out'] < s['opposed_cos_lt'])
    put('near_channel_now', K.select_top(cu['c_in_eff'], near, s['cap'], largest=True))
    copies = (cu['c_in_eff'] > s['near_channel_cin_gt']) & (cu['c_out'] > s['copies_cout_gt'])
    put('copies_now', K.select_top(cu['c_in_eff'], copies, s['cap'], largest=True))
    put('twins_now', K.select_top(cu['c_in_eff'], torch.ones_like(cu['c_out'], dtype=torch.bool), s['twins_top'], largest=True))
    lit = ((1 - cu['c_in_raw']) < cfg['literal']['cos_distance_lt']) & (cu['c_out'] < cfg['literal']['cout_lt'])
    put('literal_now', K.select_top(cu['c_in_raw'], lit, s['cap'], largest=True))
    if block == last_block:
        sets['functional'] = functional_pids
        info['functional'] = dict(selected=int(functional_pids.numel()), total=int(functional_pids.numel()), truncated=False)
    else:
        sets['functional'] = functional_pids[:0]
        info['functional'] = dict(selected=0, total=0, truncated=False, note='catalog exists only for the last block')
    sets['random'] = random_pids[block]
    info['random'] = dict(selected=int(random_pids[block].numel()), total=int(random_pids[block].numel()), truncated=False)
    union = torch.unique(torch.cat([sets[n] for n in SET_NAMES]))
    member = torch.zeros(union.numel(), dtype=torch.int32, device=union.device)
    pos = {}
    for bit, n in enumerate(SET_NAMES):
        p = torch.searchsorted(union, sets[n])
        member[p] |= (1 << bit)
        pos[n] = p
    return sets, info, union, member, pos


def head_positions(pos, sets, cfg):
    chosen, seen = [], set()
    for name in HEAD_PRIORITY:
        p = pos[name].tolist()
        if name == 'random':
            p = p[:cfg['sets']['head_random']]
        for q in p:
            if q not in seen and len(chosen) < cfg['sets']['head_cap']:
                seen.add(q); chosen.append(q)
    return chosen


# ----------------------------------------------------------------------------------------------- jeden krok
def run_step(torch, ctx, step, prev, final_sets, functional_pids, random_pids, out_dir, identity, check_all):
    cfg, device, net, taps, inp, head = ctx['cfg'], ctx['device'], ctx['net'], ctx['taps'], ctx['inp'], ctx['head']
    N = cfg['hidden']; L = cfg['blocks'] - 1
    iu0, iu1 = ctx['iu']
    W = ctx['W']
    begin = time.monotonic()
    qc = defaultdict(float)
    summary = dict(step=step, blocks=[], panels={})
    block_state = []
    # ---- geometria wag (cała populacja par)
    for k in range(cfg['blocks']):
        w = W[k]
        C_out = K.gram_cosine(w['A']); C_eff = K.cosine_matrix(w['theta_eff']); C_raw = K.cosine_matrix(w['theta_raw'])
        cu = dict(c_out=C_out[iu0, iu1], c_in_eff=C_eff[iu0, iu1], c_in_raw=C_raw[iu0, iu1])
        sets, info, union, member, pos = build_sets(torch, cu, cfg, k, L, final_sets, functional_pids, random_pids)
        I, J = iu0[union], iu1[union]
        wm = K.pair_weight_metrics(w['theta_eff'], w['theta_raw'], w['V'], I, J)
        upd = None
        if prev is not None:
            blk = ((w['theta_raw'] - prev[k]['theta_raw']) ** 2).sum() + ((w['V'] - prev[k]['V']) ** 2).sum()
            upd = K.update_fractions(prev[k]['theta_raw'], prev[k]['V'], w['theta_raw'], w['V'], I, J, blk)
        lit = K.literal_flags(wm, upd, cfg['literal'])
        npc = {key: v.cpu().numpy() for key, v in cu.items()}
        pop = dict(block=k, pairs=int(len(npc['c_out'])),
                   opposed_pairs=int((npc['c_out'] < cfg['sets']['opposed_cos_lt']).sum()),
                   with_opponent=int(((C_out < cfg['sets']['opposed_cos_lt']).sum(1) > 0).sum()),
                   spearman_cin_eff_cout=K.spearman(npc['c_in_eff'], npc['c_out']),
                   spearman_cin_raw_cout=K.spearman(npc['c_in_raw'], npc['c_out']),
                   enrichment={f'{kind}_{tau}': K.enrichment(npc['c_in_' + kind], npc['c_out'], cfg['sets']['opposed_cos_lt'], tau)
                               for kind in ('eff', 'raw') for tau in cfg['enrichment_taus']},
                   quadrants={f'cin{tau}_{side}': int(((npc['c_in_eff'] > tau) & (
                       npc['c_out'] < cfg['sets']['opposed_cos_lt'] if side == 'opposed' else npc['c_out'] > cfg['sets']['copies_cout_gt'])).sum())
                       for tau in cfg['enrichment_taus'] for side in ('opposed', 'aligned')},
                   literal_population=dict(input_near=int(((1 - npc['c_in_raw']) < cfg['literal']['cos_distance_lt']).sum()),
                                           opposite=int((npc['c_out'] < cfg['literal']['cout_lt']).sum()),
                                           both=int((((1 - npc['c_in_raw']) < cfg['literal']['cos_distance_lt'])
                                                     & (npc['c_out'] < cfg['literal']['cout_lt'])).sum())),
                   literal_tracked={n: int(v.sum()) for n, v in lit.items()},
                   max_pair_param_norm=float(wm['pair_param_norm'].max()) if union.numel() else None,
                   a_norm_median=float(torch.linalg.vector_norm(w['V'], dim=1).median()),
                   sets=info)
        hist_eff, ex, ey = K.hist2d(npc['c_in_eff'], npc['c_out'], cfg['hist_bins'])
        hist_raw, _, _ = K.hist2d(npc['c_in_raw'], npc['c_out'], cfg['hist_bins'])
        block_state.append(dict(cu=cu, union=union, member=member, pos=pos, I=I, J=J, wm=wm, upd=upd, lit=lit,
                                pop=pop, hist_eff=hist_eff, hist_raw=hist_raw, edges=ex,
                                const=K.pair_constants(w['A'], I, J), sets=sets))
    hp = head_positions(block_state[L]['pos'], block_state[L]['sets'], cfg)
    hpos = torch.as_tensor(hp, dtype=torch.long, device=device)
    fpos_list = sorted(set(q for n in FILM_SETS for q in block_state[L]['pos'][n].tolist()))
    fpos = torch.as_tensor(fpos_list, dtype=torch.long, device=device)
    # ---- dane: oba panele
    acc = {}
    common = {}
    for panel, rows in ctx['selection'].items():
        n_films = cfg['panels'][panel]['films']
        A_tok = [torch.zeros(len(block_state[k]['union']), len(K.TOKEN_SUMS), dtype=torch.float64, device=device)
                 for k in range(cfg['blocks'])]
        A_film = torch.zeros(len(fpos), n_films, len(FILM_SUMS), dtype=torch.float64, device=device)
        A_head = torch.zeros(len(hpos), len(ANCHORS), len(HEAD_SUMS), dtype=torch.float64, device=device)
        mom = [dict(G=torch.zeros(N, N, dtype=torch.float64, device=device), Sg=torch.zeros(N, dtype=torch.float64, device=device),
                    Sz=torch.zeros(N, dtype=torch.float64, device=device), Z2=torch.zeros(N, N, dtype=torch.float64, device=device),
                    n=0) for _ in range(cfg['blocks'])]
        det = defaultdict(list)
        for ii, row in enumerate(rows):
            rgb = load_image(ctx['source'], row)
            x = torch.as_tensor(rgb, device=device).permute(2, 0, 1).float()[None] / 255
            with torch.no_grad():
                raw32, _ = net(x)
            raw32 = raw32.reshape(600, 5).double()
            d = taps.d
            post = d['post'][0].double()
            with torch.no_grad():
                raw = head(post)
            qc['head'] = max(qc['head'], float((raw - raw32).abs().max()))
            for k in range(cfg['blocks']):
                w = W[k]; bs = block_state[k]
                z32, g32 = d[('z', k)][0].double(), d[('g', k)][0].double()
                xin = d[('xin', k)][0].double()
                z_re = xin @ w['W1'].T + w['b1']
                qc['fc1'] = max(qc['fc1'], float((z_re - z32).abs().max()))
                qc['gelu'] = max(qc['gelu'], float((K.gelu(z32) - g32).abs().max()))
                out64 = g32 @ w['V'] + w['b2']
                qc['fc2'] = max(qc['fc2'], float((out64 - d[('out', k)][0].double()).abs().max()))
                if k == L:
                    qc['post'] = max(qc['post'], float((d[('res', k)][0].double() + out64 - post).abs().max()))
                z, g = z32, g32
                m = mom[k]
                m['G'] += g.T @ g; m['Sg'] += g.sum(0); m['Sz'] += z.sum(0); m['Z2'] += z.T @ z; m['n'] += z.shape[0]
                thr = cfg['coactivation_fraction_of_q95'] * torch.quantile(g.abs(), 0.95, dim=0)
                mlp_norm = torch.linalg.vector_norm(out64, dim=1)
                P = len(bs['union'])
                for c0 in range(0, P, cfg['pair_chunk']):
                    sl = slice(c0, min(P, c0 + cfg['pair_chunk']))
                    const = {kk: v[sl] for kk, v in bs['const'].items()}
                    sums = K.pair_token_sums(z, g, mlp_norm, bs['I'][sl], bs['J'][sl], const, thr)
                    A_tok[k][sl] += torch.stack([sums[s] for s in K.TOKEN_SUMS], 1)
                if k == L and len(fpos):
                    const = {kk: v[fpos] for kk, v in bs['const'].items()}
                    sums = K.pair_token_sums(z, g, mlp_norm, bs['I'][fpos], bs['J'][fpos], const, thr)
                    A_film[:, row['film_index']] += torch.stack([sums[s] for s in FILM_SUMS], 1)
                if (check_all or ii == 0) and len(bs['union']):
                    qc['vector'] = max(qc['vector'], vector_check(torch, z, g, mlp_norm, w['V'], bs, thr))
            # ---- detekcja i interwencje w głowie (ostatni blok)
            anc = anchors_for(raw[:, 0].cpu().numpy(), row['gt'])
            win = anc['clean_winner']
            box = decode(torch, raw[win], win).cpu().numpy()
            gtid = ctx['C'].target_gt(box[1:], row['gt'])
            det['winner'].append(win); det['logit'].append(float(box[0]))
            det['score'].append(float(1 / (1 + np.exp(-box[0]))))
            det['iou'].append(ctx['C'].iou(box[1:], row['gt'][gtid]) if gtid is not None else np.nan)
            det['gt_present'].append(bool(len(row['gt']))); det['gt_anchor'].append(anc.get('gt_anchor', -1))
            det['film_index'].append(row['film_index']); det['ids'].append(row['id'])
            if len(hpos):
                bsL, wL = block_state[L], W[L]
                zL, gL = d[('z', L)][0].double(), d[('g', L)][0].double()
                for ai, an in enumerate(ANCHORS):
                    if an not in anc:
                        continue
                    t = anc[an]
                    y, yg = K.pair_vectors(zL[t], gL[t], wL['V'], bsL['I'][hpos], bsL['J'][hpos])
                    with torch.no_grad():
                        base = decode(torch, head(post[t][None]), t)[0]
                        abl = decode(torch, head(post[t][None] - y), t)
                        glu = decode(torch, head(post[t][None] - y + yg), t)
                    da, dg_ = abl - base, glu - base
                    A_head[:, ai] += torch.stack([torch.ones_like(da[:, 0]), (da[:, 1:] ** 2).sum(1), (dg_[:, 1:] ** 2).sum(1),
                                                  da[:, 0], da[:, 0] ** 2, dg_[:, 0], dg_[:, 0] ** 2], 1)
                if ii == 0:
                    qc['hook'] = max(qc['hook'], hook_check(torch, net, taps, x, head, post, anc['clean_winner'],
                                                            bsL['I'][hpos[:2]].tolist(), bsL['J'][hpos[:2]].tolist(), L))
        acc[panel] = dict(tok=A_tok, film=A_film, head=A_head, mom=mom)
        common[panel] = det
    for key, limit in TOL.items():
        if qc.get(key, 0.0) > limit:
            raise ValueError(f'QC failed {key}: {qc[key]} > {limit}')
    # ---- zapis
    written = {}
    for k in range(cfg['blocks']):
        bs = block_state[k]; w = W[k]
        mats = {key: v.cpu().numpy().astype(np.float32) for key, v in bs['cu'].items()}
        mats.update(a_norm=torch.linalg.vector_norm(w['V'], dim=1).cpu().numpy(),
                    theta_eff_norm=torch.linalg.vector_norm(w['theta_eff'], dim=1).cpu().numpy(),
                    theta_raw_norm=torch.linalg.vector_norm(w['theta_raw'], dim=1).cpu().numpy(),
                    b_hat=w['b_hat'].cpu().numpy(), hist_eff=bs['hist_eff'], hist_raw=bs['hist_raw'], hist_edges=bs['edges'])
        pop_data = {}
        for panel, a in acc.items():
            m = a['mom'][k]
            R = K.correlation_from_moments(m['Sz'], m['Z2'], m['n'])
            kappa, c_g, c_out, beta, e = K.energy_cancellation(m['G'], w['A'])
            mats[f'c_g_{panel}'] = c_g[iu0, iu1].cpu().numpy().astype(np.float32)
            mats[f'r_z_{panel}'] = R[iu0, iu1].cpu().numpy().astype(np.float32)
            mats[f'energy_{panel}'] = e.cpu().numpy(); mats[f'diagG_{panel}'] = torch.diagonal(m['G']).cpu().numpy()
            mats[f'mean_g_{panel}'] = (m['Sg'] / m['n']).cpu().numpy(); mats[f'mean_z_{panel}'] = (m['Sz'] / m['n']).cpu().numpy()
            ku = kappa[iu0, iu1].cpu().numpy(); cg = mats[f'c_g_{panel}']; co = bs['cu']['c_out'].cpu().numpy()
            hk, kx, ky = K.hist2d(ku, co, cfg['hist_bins'], xrange=(0.0, 2.0))
            mats[f'hist_kappa_{panel}'] = hk
            opp = co < cfg['sets']['opposed_cos_lt']
            pop_data[panel] = dict(tokens=int(m['n']),
                                   all=dict(kappa_median=float(np.nanmedian(ku)), c_g_median=float(np.nanmedian(cg))),
                                   opposed=dict(count=int(opp.sum()),
                                                kappa_median=float(np.nanmedian(ku[opp])) if opp.any() else None,
                                                kappa_lt_0_5=float((ku[opp] < .5).mean()) if opp.any() else None,
                                                c_g_median=float(np.nanmedian(cg[opp])) if opp.any() else None,
                                                r_z_median=float(np.nanmedian(mats[f'r_z_{panel}'][opp])) if opp.any() else None))
        mats['hist_kappa_edges_x'] = np.linspace(0.0, 2.0, cfg['hist_bins'] + 1)
        bs['pop']['data'] = pop_data
        ident = dict(identity, step=step, block=k)
        written[f'mat_b{k}'] = save_npz(out_dir / f'mat_b{k}.npz', dict(ident, kind='matrices'), mats, 0.0)['sha256']
        pairs = dict(pid=bs['union'].cpu().numpy().astype(np.int64), I=bs['I'].cpu().numpy().astype(np.int32),
                     J=bs['J'].cpu().numpy().astype(np.int32), membership=bs['member'].cpu().numpy(),
                     set_names=np.asarray(SET_NAMES), token_sum_names=np.asarray(K.TOKEN_SUMS),
                     random_order=random_pids[k].cpu().numpy())
        for key, v in bs['wm'].items():
            pairs['w_' + key] = v.cpu().numpy()
        if bs['upd'] is not None:
            for key, v in bs['upd'].items():
                pairs['u_' + key] = v.cpu().numpy()
        for key, v in bs['lit'].items():
            pairs['l_' + key] = v.cpu().numpy()
        for panel, a in acc.items():
            pairs['t_' + panel] = a['tok'][k].cpu().numpy()
            if k == L:
                pairs['f_' + panel] = a['film'].cpu().numpy(); pairs['h_' + panel] = a['head'].cpu().numpy()
        if k == L:
            pairs.update(f_positions=fpos.cpu().numpy(), f_sum_names=np.asarray(FILM_SUMS), h_positions=hpos.cpu().numpy(),
                         h_sum_names=np.asarray(HEAD_SUMS), h_anchor_names=np.asarray(ANCHORS))
        written[f'pairs_b{k}'] = save_npz(out_dir / f'pairs_b{k}.npz', dict(ident, kind='pairs'), pairs, 0.0)['sha256']
        summary['blocks'].append(bs['pop'])
    com = {}
    for panel, det in common.items():
        for key, v in det.items():
            com[f'{panel}_{key}'] = np.asarray(v)
        summary['panels'][panel] = dict(images=len(det['ids']),
                                        iou_mean_gt=float(np.nanmean(np.asarray(det['iou'], dtype=float))) if np.isfinite(np.asarray(det['iou'], dtype=float)).any() else None,
                                        score_mean=float(np.mean(det['score'])),
                                        score_mean_gt=float(np.mean([s for s, p in zip(det['score'], det['gt_present']) if p])) if any(det['gt_present']) else None)
    written['common'] = save_npz(out_dir / 'common.npz', dict(identity, step=step, kind='common'), com, 0.0)['sha256']
    summary.update(qc=dict(qc), tolerances=TOL, seconds=time.monotonic() - begin, files=written)
    save_json(out_dir / 'STEP_SUMMARY.json', summary)
    save_json(out_dir / 'STEP_DONE.json', dict(identity, step=step, status='complete', files=written,
                                                summary_sha256=file_sha(out_dir / 'STEP_SUMMARY.json')))
    return summary


def vector_check(torch, z, g, mlp_norm, V, bs, thr, n=64):
    """Wzory skalarne (Gram) vs jawne wektory na jednym tokenie dla do n par."""
    P = min(n, len(bs['union']))
    sel = torch.linspace(0, len(bs['union']) - 1, P, device=z.device).round().long()
    I, J = bs['I'][sel], bs['J'][sel]
    const = {k: v[sel] for k, v in bs['const'].items()}
    t = int(torch.argmax(mlp_norm))
    s = K.pair_token_sums(z[t:t + 1], g[t:t + 1], mlp_norm[t:t + 1], I, J, const, thr)
    y, yg = K.pair_vectors(z[t], g[t], V, I, J)
    e1 = (s['y2'] - (y * y).sum(1)).abs() / torch.clamp((y * y).sum(1), min=1e-12)
    e2 = (s['err2'] - ((y - yg) ** 2).sum(1)).abs() / torch.clamp((y * y).sum(1), min=1e-12)
    return float(torch.maximum(e1, e2).max())


def hook_check(torch, net, taps, x, head, post, tok, Is, Js, L):
    """Rzeczywiste zerowanie aktywacji pary na tokenie vs szybki odczyt head(post - y)."""
    worst = 0.0
    act = net.bloki[L].mlp.act
    for i, j in zip(Is, Js):
        def ablate(m, a, o, i=i, j=j):
            r = o.clone(); r[:, tok, [i, j]] = 0; return r
        h = act.register_forward_hook(ablate)
        try:
            with torch.no_grad():
                direct, _ = net(x)
        finally:
            h.remove()
        V = net.bloki[L].mlp.fc2.weight.detach().double().T
        with torch.no_grad():
            net(x)  # czysty forward: przywraca stan haków i daje oryginalne aktywacje
        g0 = taps.d[('g', L)][0].double()
        y = g0[tok, i] * V[i] + g0[tok, j] * V[j]
        with torch.no_grad():
            fast = head(post[tok][None] - y)[0]
        worst = max(worst, float((direct.reshape(600, 5)[tok].double() - fast).abs().max()))
    return worst


# ----------------------------------------------------------------------------------------------- seed
def run_seed(args):
    import torch
    cfg = read_json(HERE / 'config.json')
    exp, source, pruning = resolve_paths(cfg, args)
    freeze_sha = verify_freeze(exp) if not args.no_freeze else 'UNFROZEN-LOCAL-TEST'
    seed = args.seed if args.seed is not None else cfg['seeds'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
    if seed not in cfg['seeds']:
        raise ValueError('Unregistered seed')
    smoke = bool(args.smoke)
    pilot = args.pilot_images is not None
    if smoke and pilot:
        raise ValueError('Choose smoke or pilot')
    mode = 'smoke' if smoke else 'pilot' if pilot else 'production'
    data_root = exp / 'Wyniki' / 'Dane'
    if mode == 'production' and not args.no_freeze:
        sm = read_json(data_root / 'smoke' / 'SMOKE.json')
        if sm.get('status') != 'PASS' or sm.get('freeze_sha256') != freeze_sha:
            raise ValueError('Matching smoke must pass first')
    out_root = data_root / {'smoke': 'smoke', 'pilot': 'pilot', 'production': 'seeds'}[mode] / f's{seed}'
    if args.out_root:
        out_root = Path(args.out_root) / f's{seed}'
    steps = cfg['smoke']['steps'] if smoke else cfg['steps']
    if args.steps:
        steps = [int(s) for s in args.steps.split(',')]
        if not set(steps) <= set(cfg['steps']) or steps != sorted(steps):
            raise ValueError('Steps must be an ordered subset of config steps')
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    with seed_lock(data_root / '_blokady', ('smoke_' if smoke else '') + f's{seed}', enabled=not args.no_lock):
        complete = out_root / 'SEED_COMPLETE.json'
        if complete.exists():
            done = read_json(complete)
            for name, sha in done['files'].items():
                if file_sha(out_root / name) != sha:
                    raise ValueError('Completed seed changed: ' + name)
            print('Seed already complete and verified', flush=True)
            return
        inp = prepare_inputs(cfg, source, pruning, seed)
        sys.path.insert(1, str(source / 'code'))
        import model as M  # noqa: E402  (kod źródłowy, SHA zweryfikowane)
        import core as C  # noqa: E402
        net = M.MalyViT().to(device).eval().requires_grad_(False)
        taps = Taps(torch, net)
        N = cfg['hidden']
        iu = K.upper_indices(N, device)
        if mode == 'smoke':
            selection = {p: inp['panels'][p][:cfg['smoke']['images']] for p in cfg['smoke']['panels']}
        elif mode == 'pilot':
            # pozycja 5 każdego filmu (co druga klatka panelu), pierwsze K filmów — K różnych filmów na panel
            selection = {p: rows[0::2][:args.pilot_images] for p, rows in inp['panels'].items()}
        else:
            selection = inp['panels']
        identity = dict(experiment=cfg['experiment'], seed=seed, freeze_sha256=freeze_sha,
                        source_freeze_sha256=inp['prov']['source_freeze_sha256'], catalog_sha256=inp['prov']['catalog_sha256'],
                        mode=mode, steps=steps, images={p: len(r) for p, r in selection.items()})
        # ---- zbiory stałe: końcowe pary przeciwstawne, katalog funkcjonalny, pary losowe
        final_state, final_sha = load_state(torch, source, seed, cfg['steps'][-1], inp)
        net.load_state_dict(final_state['net'], strict=True)
        Wf = block_weights(torch, net, device)
        final_sets = {}
        for k in range(cfg['blocks']):
            c_out = K.gram_cosine(Wf[k]['A'])[iu[0], iu[1]]
            idx, total, trunc = K.select_top(c_out, c_out < cfg['sets']['opposed_cos_lt'], cfg['sets']['cap'], largest=False)
            final_sets[k] = dict(idx=idx, info=dict(selected=int(idx.numel()), total=int(total), truncated=bool(trunc),
                                                     defined_at_step=cfg['steps'][-1]))
        functional_pids = torch.as_tensor(sorted({pid_of(i, j, N) for i, j in inp['functional']}), dtype=torch.long, device=device)
        random_pids = {}
        P_all = N * (N - 1) // 2
        for k in range(cfg['blocks']):
            rng = np.random.default_rng(cfg['random_seed'] + 1000 * k + seed)
            random_pids[k] = torch.as_tensor(rng.choice(P_all, cfg['sets']['random_pairs'], replace=False), dtype=torch.long, device=device)
        ctx = dict(cfg=cfg, device=device, net=net, taps=taps, inp=inp, source=source, iu=iu, C=C, selection=selection)
        save_json(out_root / 'SEED_PROVENANCE.json', dict(identity, provenance=inp['prov'], device=device,
                  torch=torch.__version__, python=platform.python_version(), final_checkpoint_sha256=final_sha,
                  final_sets={k: v['info'] for k, v in final_sets.items()}, functional_pairs=len(inp['functional']),
                  panels={p: [r['id'] for r in rows] for p, rows in selection.items()}))
        prev = None
        for step in steps:
            state, sha = load_state(torch, source, seed, step, inp)
            geometry_check(state['net'], inp['geometry']['checkpoints'][str(step)]['blocks'])
            net.load_state_dict(state['net'], strict=True)
            W = block_weights(torch, net, device)
            ctx['W'] = W
            import copy
            ctx['head'] = copy.deepcopy(torch.nn.Sequential(net.norm, net.glowa)).double()
            out_dir = out_root / f'k{step:05d}'
            sd = out_dir / 'STEP_DONE.json'
            if sd.exists():
                rec = read_json(sd)
                if rec.get('freeze_sha256') != freeze_sha or rec.get('checkpoint_sha256') != sha:
                    raise ValueError('Completed step identity differs')
                for name, sha_f in rec['files'].items():
                    if file_sha(out_dir / (name + '.npz')) != sha_f:
                        raise ValueError('Completed shard changed: ' + name)
                print(f'seed {seed} k{step}: verified, skipped', flush=True)
            else:
                t0 = time.monotonic()
                s = run_step(torch, ctx, step, prev, final_sets, functional_pids, random_pids, out_dir,
                             dict(identity, checkpoint_sha256=sha), check_all=smoke)
                print(f'seed {seed} k{step}: {time.monotonic() - t0:.1f}s qc={dict(s["qc"])}', flush=True)
            prev = [dict(theta_raw=w['theta_raw'], V=w['V']) for w in W]
        files = {p.relative_to(out_root).as_posix(): file_sha(p) for p in sorted(out_root.rglob('*'))
                 if p.is_file() and not p.name.endswith('.tmp') and p.name != 'SEED_COMPLETE.json'}
        save_json(complete, dict(identity, status='complete', steps=steps, files=files, device=device,
                                 gpu=torch.cuda.get_device_name() if device == 'cuda' else None, torch=torch.__version__))
        if smoke:
            save_json(data_root / 'smoke' / 'SMOKE.json', dict(status='PASS', freeze_sha256=freeze_sha, seed=seed,
                      steps=steps, seed_complete_sha256=file_sha(complete), device=device, torch=torch.__version__,
                      gpu=torch.cuda.get_device_name() if device == 'cuda' else None))
        print('done', seed, flush=True)


def load_state(torch, source, seed, step, inp):
    path = checkpoint_path(source, seed, step)
    sha = file_sha(path)
    key = f'replications/s{seed}/weights/k{step:05d}.pt'
    if sha != inp['weights'][key] or sha != inp['geometry']['checkpoints'][str(step)]['checkpoint_sha256']:
        raise ValueError('Checkpoint SHA differs: ' + key)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['seed'] != seed or state['krok'] != step:
        raise ValueError('Checkpoint identity differs')
    return state, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--exp-root'); p.add_argument('--source-root'); p.add_argument('--pruning-root'); p.add_argument('--out-root')
    p.add_argument('--device'); p.add_argument('--no-lock', action='store_true')
    p.add_argument('--pilot-images', type=int, help='Pilot: K klatek na panel (pozycja 5 w K pierwszych filmach)')
    p.add_argument('--steps', help='Podzbiór kroków, np. 0,1000,10000,65640')
    p.add_argument('--no-freeze', action='store_true', help='Tylko lokalne testy przed zamrożeniem; wynik nie jest produkcyjny')
    run_seed(p.parse_args())


if __name__ == '__main__':
    main()
