"""Pole sił w kuli komórki wybranego tokenu w każdym z 12 podkroków jednego przejścia przez sieć (okienko 11, podkroki).

Układ jak przejście przez sieć (history.podkroki, układ '_n'): położenie p = (x − x̄)/s_ref rzutowane wspólnym rzutem P (3 × 192),
s_ref = średnia norma tokenu po 13 stanach. Próbna cząsteczka stoi w x_j + s_ref Pᵀ u (u w jednostkach '_n'), reszta tokenów bez zmian.
  MLP bloku b (stan 2b+1):  f(u) = P [(z(x_j + δ) − z̄) W₂ᵀ] / s_ref,  z = GELU(fc1(LN₂ x)),  z̄ = średnia po tokenach
  uwaga bloku b (stan 2b):  f(u) = P [Σ_h (A^h(x_j + δ) − ā^h) U^h] / s_ref,  A^h = softmax wiersza słuchacza (klucze i wartości
                            prawdziwych tokenów), ā^h_k = średnia po słuchających, U^h_k = W_o[:, h] v^h_k
Siły są względem średniej po tokenach jak w przejściu (biasy znikają), więc f(0) = dokładny ruch tokenu j w tym podkroku
(kontrola: blad). Komórka: kula o promieniu r_kom = połowa odległości do najbliższego innego tokenu w stanie przed podkrokiem;
siatka G³ na sześcianie [−L, L]³, L = zasięg · r_kom (okienko pokazuje tylko kulę o promieniu L). Kolor węzła: trzy największe wkłady (lasery: kolory
reflektorów bloku, uwaga: kolory cząsteczek-źródeł). Wynik w postaci jak focal_trap.policz (K = 12 podkroków), do FocalField.
"""
import numpy as np

from laser_app.compute import history as hist, passage_store


def policz(wagi, krok, rgb, j, d, G=17, kol_cz=None, urzadzenie='cuda', zasieg=1.0, st=None):
    """d: przejście tego checkpointu (history.podkroki albo z dysku, z 'baza'); j: token; kol_cz (600, 3) kolory cząsteczek."""
    import torch
    import torch.nn.functional as Fn
    from pathlib import Path
    from laser_app.compute.common import kolory_kanalow
    dev = torch.device(urzadzenie if torch.cuda.is_available() else 'cpu')
    M = hist.modul_modelu()
    j = int(j); G = int(G)
    pos = np.asarray(d['pos_n'], np.float64)                                        # (13, 600, 3)
    P = torch.as_tensor(np.asarray(d['baza']['P_n']), dtype=torch.float32, device=dev)   # (3, 192)
    s_ref = float(np.asarray(d['rms'], np.float64).mean())
    kol_k = 0.25 + 0.75 * kolory_kanalow()[:, :3]
    kol_cz = np.full((hist.TOK, 3), 0.8) if kol_cz is None else np.asarray(kol_cz, np.float64)[:, :3]
    lin = torch.linspace(-1.0, 1.0, G, device=dev)
    gz, gy, gx = torch.meshgrid(lin, lin, lin, indexing='ij')
    SIATKA = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], 1)      # indeks i = ix + G·iy + G²·iz (jak shader)
    N = G ** 3
    POLE = np.zeros((12, N, 4), np.float32); KOLOR = np.zeros((12, N, 4), np.float32)
    L = np.zeros(12, np.float32); SKOK = np.zeros((12, 3), np.float32); FMED = np.zeros(12, np.float32)
    BLAD = np.zeros(12, np.float32)
    with passage_store.ZAMEK_SIECI:                                                           # wspólna sieć history._siec
        if dev.type not in hist._siec:
            hist._siec[dev.type] = M.MalyViT().to(dev).eval().requires_grad_(False)
        net = hist._siec[dev.type]
        if st is None:                                                             # st: wagi wczytane wcześniej (pomiar historii)
            st = torch.load(Path(wagi) / f'k{int(krok):05d}.pt', map_location=dev, weights_only=True)
        net.load_state_dict(st.get('net', st), strict=True)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255).to(dev)
            f = (x - M.MEAN.to(dev)) / M.STD.to(dev)
            for modul in net.lodyga:
                f = modul(f)
            B_, C, h, w = f.shape
            t = f.flatten(2).transpose(1, 2) + M.sincos_2d(h, w, C, dev)
            for b, blk in enumerate(net.bloki):
                for faza in (0, 1):                                                 # 0 uwaga (stan 2b), 1 MLP (stan 2b + 1)
                    k = 2 * b + faza
                    X = t[0]                                                        # (600, 192) stan przed podkrokiem
                    odl = np.linalg.norm(pos[k] - pos[k, j], axis=1); odl[j] = np.inf
                    rk = float(zasieg) * 0.5 * float(odl.min()); L[k] = rk
                    U = SIATKA * rk                                                 # (N, 3) w jednostkach '_n'
                    Xp = X[j] + (U * s_ref) @ P                                     # próbne położenia (N, 192)
                    if faza == 1:
                        Z = blk.mlp.act(blk.mlp.fc1(blk.norm2(X))); zbar = Z.mean(0)
                        Zp = blk.mlp.act(blk.mlp.fc1(blk.norm2(Xp))) - zbar         # (N, 768)
                        W2 = blk.mlp.fc2.weight                                     # (192, 768)
                        F192 = Zp @ W2.T
                        nv = (W2.T @ P.T).norm(dim=1)                               # |P v_c|
                        wk = Zp.abs() * nv; kol = kol_k[d['slot'][b]]
                    else:
                        at = blk.attn; H = at.num_heads; hd = C // H
                        def qkv_(Y):
                            q_ = at.qkv(blk.norm1(Y)).reshape(Y.shape[0], 3, H, hd).permute(1, 2, 0, 3)   # (3, H, n, hd)
                            return at.q_norm(q_[0]), at.k_norm(q_[1]), q_[2]
                        q, kk, v = qkv_(X)
                        A = ((q * at.scale) @ kk.transpose(-2, -1)).softmax(-1)          # (H, 600, 600) wiersz = słuchający
                        abar = A.mean(1)                                            # (H, 600) średnia po słuchających
                        Wo = at.proj.weight
                        Uh = torch.stack([v[g] @ Wo[:, g * hd:(g + 1) * hd].T for g in range(H)])   # (H, 600, 192)
                        qp, _, _ = qkv_(Xp)
                        Ap = ((qp * at.scale) @ kk.transpose(-2, -1)).softmax(-1) - abar[:, None, :]   # (H, N, 600)
                        F192 = torch.einsum('hnk,hkd->nd', Ap, Uh)
                        wk = torch.einsum('hnk,hk->nk', Ap.abs(), (Uh @ P.T).norm(dim=2)); kol = kol_cz
                    Fp = (F192 @ P.T) / s_ref                                       # (N, 3) w jednostkach '_n'
                    tw, ti = wk.topk(3, dim=1)
                    kol_t = torch.as_tensor(np.asarray(kol, np.float32), device=dev)
                    mix = (tw[..., None] * kol_t[ti]).sum(1) / tw.sum(1, keepdim=True).clamp_min(1e-30)
                    fm = Fp.norm(dim=1)
                    POLE[k, :, :3] = Fp.cpu().numpy(); POLE[k, :, 3] = fm.cpu().numpy()
                    KOLOR[k, :, :3] = mix.cpu().numpy(); KOLOR[k, :, 3] = (tw[:, 0] / wk.sum(1).clamp_min(1e-30)).cpu().numpy()
                    FMED[k] = float(fm.median())
                    srodek_i = N // 2                                               # węzeł w środku (G nieparzyste): u = 0
                    SKOK[k] = POLE[k, srodek_i, :3]
                    ruch = pos[k + 1, j] - pos[k, j]
                    BLAD[k] = float(np.linalg.norm(SKOK[k] - ruch) / max(np.linalg.norm(ruch), 1e-30))
                    t = t + (blk.ls1(at(blk.norm1(t))) if faza == 0 else blk.ls2(blk.mlp(blk.norm2(t))))
    sr = pos[:12, j].astype(np.float32)
    return dict(G=G, kroki=np.arange(12), pole=POLE, kolor=KOLOR, srodek=sr, srodek_siatki=sr, ogn=sr, ogn_lin=sr,
                zb=np.zeros(12, bool), skok=SKOK, L=L, fmed=FMED, r99=np.ones(12, np.float32), blad=BLAD, krok=int(krok), tok=j)
