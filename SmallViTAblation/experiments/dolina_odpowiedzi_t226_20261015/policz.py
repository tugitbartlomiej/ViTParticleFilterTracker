"""Rzeczywisty wycinek zbioru stałej odpowiedzi zamrożonej głowy, token 226.

python -X utf8 experiments/dolina_odpowiedzi_t226_20261015/policz.py
"""
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/dolina_odpowiedzi_t226_20261015'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(4)
DT = torch.float64
T0 = time.time()


def log(*a):
    print(f'[{time.time()-T0:.1f}s]', *a, flush=True)


def main():
    ckpt = ROOT / 'assets/MalyViT_35000_od_zera_s47/wagi/k65640.pt'
    history = ROOT / 'app/cache_uczenia/lasery/eafb5a973be3d490/dane.npz'
    hist = np.load(history)
    old = np.load(ROOT / 'outputs/pulapka_cel_koloidu_20261015/pomiar.npz')
    x0 = torch.tensor(old['X_wyb'][-1, list(old['wybrane']).index(226)], device=DEVICE, dtype=DT)
    st = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    st = st.get('net', st)
    w = {k: st[k].to(dtype=DT) for k in ('norm.weight', 'norm.bias', 'glowa.0.weight', 'glowa.0.bias', 'glowa.2.weight', 'glowa.2.bias')}
    eps = 1e-5  # nn.LayerNorm(192) w model.py; nie epsilon LN bloków timm.

    def head(x):
        z = F.layer_norm(x, (192,), w['norm.weight'], w['norm.bias'], eps)
        return F.linear(F.gelu(F.linear(z, w['glowa.0.weight'], w['glowa.0.bias'])), w['glowa.2.weight'], w['glowa.2.bias'])

    def jac(x):
        # Analityczny dokładny jakobian końcowej LayerNorm + MLP.
        x = torch.atleast_2d(x)
        u = x-x.mean(-1, keepdim=True)
        sd = torch.sqrt(u.square().mean(-1, keepdim=True)+eps)
        jl = (torch.eye(192, device=DEVICE, dtype=DT)-1/192)[None]/sd[..., None]
        jl = jl-u[..., :, None]*u[..., None, :]/(192*sd[..., None]**3)
        jl = w['norm.weight'][None, :, None]*jl
        z = u/sd*w['norm.weight']+w['norm.bias']
        a = F.linear(z, w['glowa.0.weight'], w['glowa.0.bias'])
        dg = .5*(1+torch.erf(a/math.sqrt(2)))+a*torch.exp(-a*a/2)/math.sqrt(2*math.pi)
        jmlp = (w['glowa.2.weight'][None]*dg[:, None]) @ w['glowa.0.weight']
        return jmlp @ jl

    gt = hist['gt'].astype(float)
    cx, cy = (226 % 30+.5)*16, (226//30+.5)*16
    target_box = np.array([cx-gt[0], cy-gt[1], gt[2]-cx, gt[3]-cy])/16
    target = torch.tensor([math.log(.995/.005), *np.log(target_box)], dtype=DT, device=DEVICE)

    def measures(x):
        o = head(x).detach().cpu().numpy()
        b = np.exp(o[..., 1:])
        inter = (np.minimum(b[..., 0], target_box[0])+np.minimum(b[..., 2], target_box[2]))*(np.minimum(b[..., 1], target_box[1])+np.minimum(b[..., 3], target_box[3]))
        area = (b[..., 0]+b[..., 2])*(b[..., 1]+b[..., 3])
        a_gt = (target_box[0]+target_box[2])*(target_box[1]+target_box[3])
        iou = inter/(area+a_gt-inter)
        E = (o[..., 0]-float(target[0]))**2+(((b-target_box)/target_box)**2).sum(-1)
        return o, expit(o[..., 0]), iou, E

    # Damped Newton, minimum-norm increments, with line search.
    xs = x0.clone()
    for it in range(90):
        er = head(xs)-target
        if er.abs().max() < 2e-12:
            break
        j = jac(xs)[0]
        dx = j.T @ torch.linalg.solve(j @ j.T+1e-9*torch.eye(5, device=DEVICE, dtype=DT), er)
        fac = 1.
        while fac > 1e-6 and (head(xs-fac*dx)-target).square().sum() >= er.square().sum():
            fac *= .5
        xs -= fac*dx
    assert float((head(xs)-target).abs().max()) < 1e-8, 'Nie znaleziono środka powierzchni'
    log('Środek', 'przesunięcie', float((xs-x0).norm()), 'miary', [v.tolist() for v in measures(xs)[1:3]])

    j = jac(xs)[0]
    j_auto = torch.autograd.functional.jacobian(head, xs)
    jac_error = float((j-j_auto).abs().max())
    assert jac_error < 1e-10
    U, sv, Vh = torch.linalg.svd(j, full_matrices=True)
    rank = int((sv > sv[0]*1e-6).sum())
    assert rank == 5
    Q = Vh[:5].T.contiguous()
    N = Vh[5:].T.contiguous()
    # Normalna miesza kierunek pewności i najbardziej czuły kierunek odczytu.
    # Dzięki temu widać zmianę pewności ORAZ ramki, w rzeczywistych proporcjach.
    e = torch.tensor([1., 0, 0, 0, 0], device=DEVICE, dtype=DT)
    a = torch.linalg.solve(j@j.T, e)
    normal = j.T@a
    normal /= normal.norm()
    principal = Vh[0] * torch.sign(j[0]@Vh[0])
    normal = .65*normal+.75*principal
    normal /= normal.norm()
    # Dwa kierunki z największą BEZWZGLĘDNĄ krzywizną kompensacji normalnej.
    coeff = torch.linalg.solve(j@j.T, j@normal)
    hess = torch.autograd.functional.hessian(lambda z: head(z)@coeff, xs)
    curv, ev = torch.linalg.eigh(N.T@hess@N)
    order = torch.argsort(curv.abs(), descending=True)
    N = N@ev[:, order]
    t1, t2 = N[:, 0], N[:, 1]
    null_residual = (j@N).norm(dim=0).cpu().numpy()
    assert null_residual.max() < 1e-10
    eps_fd = 1e-4
    fd = (head(xs+eps_fd*t1)-head(xs-eps_fd*t1))/(2*eps_fd)
    assert float(fd.norm()) < 1e-7
    nq = Q.T@normal
    jq_center = j@Q

    def project(base, iterations=60):
        """Ruch korekcyjny wyłącznie w 5 stałych kierunkach Q, 5 równań.

        Zmienny J@Q, batch z line search; odrzuca rozbieżne punkty.
        """
        correction = torch.zeros((len(base), 5), device=DEVICE, dtype=DT)
        for k in range(iterations):
            x = base+correction@Q.T
            er = head(x)-target
            if er.abs().max() < 1e-10:
                break
            jq = jac(x)@Q
            dc = torch.linalg.solve(jq+1e-10*torch.eye(5, device=DEVICE, dtype=DT), er[..., None])[..., 0]
            length = dc.norm(dim=1, keepdim=True)
            dc *= torch.clamp(3/length.clamp_min(1e-12), max=1)
            score = er.square().sum(-1)
            best = correction.clone()
            for fac in (1., .5, .25, .125, .0625):
                cand = correction-fac*dc
                cost = (head(base+cand@Q.T)-target).square().sum(-1)
                improve = cost < score
                best[improve] = cand[improve]
                score = torch.minimum(cost, score)
            correction = best
        x = base+correction@Q.T
        error = (head(x)-target).abs().max(-1).values
        return x, correction, error

    ng = 33
    R = 5.
    for attempt in range(6):
        uv = np.linspace(-R, R, ng)
        uu, vv = np.meshgrid(uv, uv)
        base = xs+torch.tensor(uu.ravel(), device=DEVICE)[:, None]*t1+torch.tensor(vv.ravel(), device=DEVICE)[:, None]*t2
        sheet, corr, residual = project(base)
        log('Powierzchnia', ng*ng, 'promień wycinka', R, 'maks. reszta', float(residual.max()))
        if float(residual.max()) < 1e-7:
            break
        R *= .7  # nie ekstrapolować lokalnego wykresu przez utratę zbieżności
    assert float(residual.max()) < 1e-7
    sheet_z = (corr@nq).cpu().numpy().reshape(ng, ng)
    so, sp, si, se = measures(sheet)

    # Zamknięta trasa po powierzchni, z przystankiem na starcie w środku.
    ph = np.linspace(0, 2*np.pi, 361)
    path_u = .76*R*np.sin(ph)
    path_v = .64*R*np.sin(2*ph)
    base = xs+torch.tensor(path_u, device=DEVICE)[:, None]*t1+torch.tensor(path_v, device=DEVICE)[:, None]*t2
    path, pcorr, pres = project(base)
    po, pp, pi, pe = measures(path)
    path_xyz = np.column_stack([path_u, path_v, (pcorr@nq).cpu().numpy()])
    assert float(pres.max()) < 1e-7

    # Zawrócenie wzdłuż prostej stycznej ujawnia krzywiznę.
    straight_s = np.linspace(-R, R, 201)
    straight = xs+torch.tensor(straight_s, device=DEVICE)[:, None]*t1
    sto, stp, sti, ste = measures(straight)
    # Normalna w środku; logit rośnie do dodatniej strony, ramka też może zmieniać się nieliniowo.
    ns = np.linspace(-12, 12, 241)
    normal_x = xs+torch.tensor(ns, device=DEVICE)[:, None]*normal
    no, nprob, ni, ne = measures(normal_x)

    # Wycinek rodziny 3-parametrowej: wszystkie miary policzone w 192D.
    delta = np.linspace(-8, 8, 33)
    chunks, volume_o = [], []
    for ds in delta:
        mo = measures(sheet+float(ds)*normal)
        chunks.append(np.column_stack(mo[1:]))
        volume_o.append(mo[0])
    volume_m = np.stack(chunks)  # delta, punkt, p/iou/E
    vol_xyz = np.stack([np.column_stack([uu.ravel(), vv.ravel(), sheet_z.ravel()+ds]) for ds in delta])
    # Dolina niezgodności: u i delta; v=0; wysokość=E, nie położenie.
    u_energy = np.linspace(-R, R, 65)
    base = xs+torch.tensor(u_energy, device=DEVICE)[:, None]*t1
    bottom, bcorr, bres = project(base)
    en_delta = np.linspace(-8, 8, 65)
    energy_results = [measures(bottom+float(ds)*normal) for ds in en_delta]
    energy_m = np.stack([np.column_stack(r[1:]) for r in energy_results])

    # Weryfikacja implementacji względem rzeczywistych modułów torch, bez ładowania reszty ViT.
    ln = torch.nn.LayerNorm(192, eps=eps, device=DEVICE, dtype=DT)
    mlp = torch.nn.Sequential(torch.nn.Linear(192, 192, device=DEVICE, dtype=DT), torch.nn.GELU(), torch.nn.Linear(192, 5, device=DEVICE, dtype=DT))
    ln.load_state_dict({'weight': w['norm.weight'], 'bias': w['norm.bias']})
    mlp.load_state_dict({k.removeprefix('glowa.'): v for k,v in w.items() if k.startswith('glowa.')})
    with torch.no_grad():
        module_error = float((mlp(ln(sheet[:17]))-head(sheet[:17])).abs().max())
    assert module_error < 1e-12
    log('Przekrój normalny: p', float(nprob.min()), float(nprob.max()), 'min IoU', float(ni.min()), 'max E', float(ne.max()))
    assert np.ptp(nprob) > 1e-4 and float(ne.max()) > 1e-3
    assert float(sti.min()) < .999 and np.ptp(sheet_z) > .05, 'Wybrany wycinek nie pokazuje mierzalnej krzywizny'
    assert np.max(np.abs(sp-.995)) < 1e-8 and si.min() > 1-1e-7
    vals = dict(x0=x0.cpu().numpy(), xstar=xs.cpu().numpy(), target=target.cpu().numpy(), target_box=target_box,
                gt=gt, rgb=hist['rgb'], Q=Q.cpu().numpy(), N=N.cpu().numpy(), normal=normal.cpu().numpy(),
                J=j.cpu().numpy(), singular_values=sv.cpu().numpy(), null_residual=null_residual,
                uv=uv, sheet_x=sheet.cpu().numpy().reshape(ng,ng,192), sheet_xyz=np.stack([uu,vv,sheet_z],-1),
                sheet_o=so.reshape(ng,ng,5), sheet_p=sp.reshape(ng,ng), sheet_iou=si.reshape(ng,ng), sheet_error=residual.cpu().numpy().reshape(ng,ng),
                path_x=path.cpu().numpy(), path_xyz=path_xyz, path_o=po, path_p=pp, path_iou=pi, path_E=pe,
                normal_s=ns, normal_o=no, normal_p=nprob, normal_iou=ni, normal_E=ne,
                straight_s=straight_s, straight_o=sto, straight_p=stp, straight_iou=sti, straight_E=ste,
                volume_xyz=vol_xyz, volume_metrics=volume_m, volume_o=np.stack(volume_o), volume_delta=delta,
                energy_u=u_energy, energy_delta=en_delta, energy_metrics=energy_m, energy_o=np.stack([r[0] for r in energy_results]),
                baseline_o=measures(x0)[0], baseline_p=measures(x0)[1], baseline_iou=measures(x0)[2],
                curvature_eigenvalues=curv[order].cpu().numpy())
    np.savez_compressed(OUT/'geometria.npz', **vals)
    good = (volume_m[...,0]>=.99)&(volume_m[...,1]>=.9)
    report = dict(checkpoint=str(ckpt), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(), token=226, step=65640,
                  state_dimension=192, head_output_dimension=5, layernorm_epsilon=eps, rank=rank, null_dimension=192-rank,
                  target_probability=.995, target_ltrb=target_box.tolist(), display_probability_threshold=.99, display_iou_threshold=.9,
                  baseline_probability=float(vals['baseline_p']), baseline_iou=float(vals['baseline_iou']),
                  constructed_shift=float((xs-x0).norm()), constructed_shift_relative=float((xs-x0).norm()/x0.norm()),
                  singular_values=sv.cpu().tolist(), max_JN_norm=float(null_residual.max()), analytic_autograd_error=jac_error,
                  centered_difference_null_norm=float(fd.norm()), module_error=module_error,
                  sheet_max_abs_output_error=float(residual.max()), path_max_abs_output_error=float(pres.max()),
                  sheet_p_range=[float(sp.min()),float(sp.max())], sheet_min_iou=float(si.min()),
                  normal_p_range=[float(nprob.min()),float(nprob.max())], normal_min_iou=float(ni.min()),
                  straight_min_iou=float(sti.min()), straight_max_energy=float(ste.max()),
                  projected_sheet_height_range=[float(sheet_z.min()),float(sheet_z.max())],
                  surface_tangent_coordinate_radius=R,
                  good_samples=int(good.sum()), total_volume_samples=int(good.size),
                  surface_definition='x*=p0.995, GT box; x(u,v)=x*+u*t1+v*t2+Q*c(u,v); head(x)=target; t1,t2 from curvature eigensystem in ker J(x*)',
                  volume_definition='x(u,v,delta)=x(u,v)+delta*n; 3-param family, not full192D volume',
                  energy_definition='(logit-logit*)^2 + sum(((ltrb-ltrb*)/ltrb*)^2); visualization discrepancy, not training loss',
                  elapsed_seconds=time.time()-T0)
    (OUT/'POMIAR.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    log(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
