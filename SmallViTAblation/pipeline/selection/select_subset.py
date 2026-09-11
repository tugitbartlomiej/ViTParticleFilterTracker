"""JEDEN skrypt selekcji podzbiorow dla wszystkich 15 kombinacji macierzy.

Wejscie:  queue.jsonl (wiersz = bieg: run_id, variant, groups, weights, seed,
          budget, arch, opcjonalnie override), cache cech, protocol.yaml.
Wyjscie:  variants/<run_id>.json dokladnie wg agents/README.md sekcja 4.2.

Algorytm (AGENT_1 zadanie 4):
  1. grupy i WAGI z queue.jsonl (nie liczymy wlasnych),
  2. kazda grupe standaryzujemy osobno (mean 0, std 1; std==0 -> zera),
  3. DINO -> PCA do features.dino_pca_dim,
  4. sklejamy bloki w stalej kolejnosci [dino, el2n, sam, fourier],
     kazdy blok mnozymy przez wage grupy,
  5. grupujemy features.cluster_algorithm (kmeans | kcenter_greedy),
     k = budget, ziarno = seed,
  6. przedstawiciel klastra wg features.cluster_representative
     (centroid | medoid | random_in_cluster).

Twarde: tylko split=='train'; len(frame_ids) == budget DOKLADNIE;
ten sam seed => identyczny wynik co do klatki.

Regula dla PUSTYCH klastrow (patrz tez README.md tego katalogu):
  kmeans moze zwrocic < budget niepustych klastrow. Brakujacych
  przedstawicieli dobieramy deterministycznie: z klatek jeszcze niewybranych,
  posortowanych rosnaco po frame_id, bierzemy pierwsze wolne, az do budzetu.

Override'y z queue.jsonl (biegi wrazliwosci) sa obsluzone: dino_pca_dim,
cluster_algorithm, cluster_representative, group_weight_delta(+target_group).

Uzycie:
    py -3.11 pipeline/selection/select_subset.py --queue queue.jsonl
        [--run-id X] [--cache-dir DIR] [--variants-dir DIR]

Kuracja Cataract-LMM (oddzielne profile i pomiary, bez historycznego DESF):
    python pipeline/selection/select_subset.py --profile balanced --subset tracking
        --budget 2500 --out curated/tracking
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Dispatch before importing the legacy feature-cache contract. The new profiles
# also work without that optional historical pipeline installed/configured.
if __name__ == "__main__" and any(
        arg == "--profile" or arg.startswith("--profile=") for arg in sys.argv[1:]):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from SmallViTAblation.pipeline.curation.run import main as curate_main
    curate_main()
    raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from common import (ROOT, gather_rows, load_feature_pkl, load_protocol,
                    read_manifest, resolve_under_root, train_pool)

GROUP_KEY = {"D": "dino", "E": "el2n", "S": "sam", "F": "fourier"}
GROUP_ORDER = ("dino", "el2n", "sam", "fourier")  # stala kolejnosc sklejenia


def load_group_matrix(name: str, pool_frame_ids: list[str], cache_dir: Path,
                      seed: int) -> tuple[np.ndarray, dict]:
    if name == "el2n":
        path = cache_dir / f"el2n_{seed}.pkl"
    else:
        path = cache_dir / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Brak cache '{path}'. Najpierw policz cechy "
            f"(build_cache.py / el2n_from_proxy.py --seed {seed}).")
    cache = load_feature_pkl(path)
    return gather_rows(cache, pool_frame_ids, str(path)), cache.get("meta", {})


def standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    out = (x - mu) / sd_safe
    out = np.where(sd > 1e-12, out, 0.0)  # kolumny stale -> zera
    return out.astype(np.float32)


def pca_dino(x: np.ndarray, dim: int) -> tuple[np.ndarray, int]:
    from sklearn.decomposition import PCA
    n_comp = int(min(dim, x.shape[1], x.shape[0]))
    if n_comp >= x.shape[1]:
        return x, x.shape[1]
    # svd_solver='full' -> wynik w pelni deterministyczny (bez SVD losowego)
    p = PCA(n_components=n_comp, svd_solver="full")
    return p.fit_transform(x).astype(np.float32), n_comp


def apply_overrides(weights: dict, groups: list[str], cfg: dict,
                    override: dict | None) -> tuple[dict, dict]:
    """Zwraca (wagi_aktywne, cfg_po_override). Wagi bazowo z queue.jsonl."""
    cfg = dict(cfg)
    w = {g: float(weights.get(g, 0.0)) for g in GROUP_ORDER}
    if override:
        for key in ("dino_pca_dim", "cluster_algorithm",
                    "cluster_representative"):
            if key in override:
                cfg[key] = override[key]
        if "group_weight_delta" in override:
            delta = float(override["group_weight_delta"])
            target = GROUP_KEY.get(override.get("target_group", ""), None)
            if target and target in GROUP_ORDER:
                w[target] = max(0.0, w[target] + delta)
                total = sum(w.values())
                if total > 0:  # renormalizacja proporcjonalna do sumy 1
                    w = {k: v / total for k, v in w.items()}
    return w, cfg


def cluster_kmeans(x: np.ndarray, k: int, seed: int, max_iter: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                max_iter=max_iter, random_state=seed)
    labels = km.fit_predict(x)
    return labels, km.cluster_centers_


def cluster_kcenter_greedy(x: np.ndarray, k: int, *, return_centers=False) -> np.ndarray:
    """K-center greedy (farthest point). Deterministyczny, bez ziarna.

    return_centers=True zwraca faktycznie wybrane indeksy. Wariant etykiet
    zachowano dla konsumentow potrzebujacych partycji Voronoia.
    """
    n = x.shape[0]
    if not 1 <= k <= n or not np.isfinite(x).all():
        raise ValueError("k-center wymaga 1 <= k <= N i skonczonych cech")
    mu = x.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(x - mu, axis=1)))
    chosen = [first]
    dist = np.linalg.norm(x - x[first], axis=1)
    dist[first] = -np.inf
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(x - x[nxt], axis=1))
        dist[chosen] = -np.inf  # identyczne wektory nadal wybieramy tylko raz
    chosen_arr = np.array(chosen, dtype=np.int64)
    if return_centers:
        return chosen_arr
    labels = np.zeros(n, dtype=np.int64)
    # O(chunk*K) pamieci zamiast tensora O(chunk*K*D).
    centres = x[chosen_arr]
    norms = np.sum(centres**2, axis=1)
    chunk_size = max(1, min(4096, 1_000_000 // k))
    for s in range(0, n, chunk_size):
        part = x[s:s+chunk_size]
        squared = np.sum(part**2, axis=1)[:, None] + norms[None, :] - 2*part @ centres.T
        labels[s:s+len(part)] = squared.argmin(axis=1)
    labels[chosen_arr] = np.arange(k)  # takze przy dokladnie identycznych centrach
    return labels


def pick_representatives(x: np.ndarray, labels: np.ndarray, k: int,
                         rule: str, seed: int) -> list[int]:
    """Po jednym indeksie z klastra; puste klastry -> regula z README."""
    chosen: list[int] = []
    taken: set[int] = set()
    rng = np.random.RandomState(seed)
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue  # pusty klaster - uzupelnimy nizej
        idx_sorted = np.sort(idx)
        if rule == "centroid":
            # punkt najblizszy srodkowi klastra; tie-break: mniejszy indeks
            d = np.linalg.norm(x[idx] - x[idx].mean(axis=0), axis=1)
            order = np.lexsort((idx, d))
            pick = int(idx[order[0]])
        elif rule == "medoid":
            sub = x[idx]
            if idx.size > 1500:  # ochrona przed O(n^2)
                d = np.linalg.norm(sub - sub.mean(axis=0), axis=1)
            else:
                d = np.linalg.norm(sub[:, None, :] - sub[None, :, :],
                                   axis=2).sum(axis=1)
            order = np.lexsort((idx, d))
            pick = int(idx[order[0]])
        elif rule == "random_in_cluster":
            pick = int(idx_sorted[rng.randint(len(idx_sorted))])
        else:
            raise ValueError(f"Nieznana regula representative: {rule}")
        if pick in taken:  # nie powinno sie zdarzyc (klastry rozlaczne)
            for alt in idx_sorted:
                if int(alt) not in taken:
                    pick = int(alt)
                    break
            else:
                continue
        taken.add(pick)
        chosen.append(pick)

    if len(chosen) < k:
        # DETERMINISTYCZNA regula dla pustych klastrow (README):
        # pierwsze wolne klatki w kolejnosci rosnacego indeksu puli
        # (pula jest posortowana po frame_id, wiec = rosnacy frame_id).
        for i in range(x.shape[0]):
            if len(chosen) >= k:
                break
            if i not in taken:
                taken.add(i)
                chosen.append(i)
    return chosen


def select_one(run: dict, pool: list[dict], pool_ids: list[str],
               cache_dir: Path, feat_cfg: dict) -> dict:
    run_id = run["run_id"]
    seed = int(run["seed"])
    budget = run["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"{run_id}: budget musi byc int (dostalem {budget!r})")
    groups = list(run.get("groups") or [])
    if not groups:
        raise ValueError(f"{run_id}: pusta lista groups - to bieg baseline? "
                         f"Uzyj baselines.py")
    weights, cfg = apply_overrides(run.get("weights") or {}, groups,
                                   dict(feat_cfg), run.get("override"))
    nieznane = [g for g in groups if g not in GROUP_KEY]
    if nieznane:
        raise ValueError(f"{run_id}: nieznane grupy {nieznane}; "
                         f"dozwolone: {sorted(GROUP_KEY)}")
    active = [GROUP_KEY[g] for g in groups]
    n = len(pool_ids)
    if budget > n:
        raise ValueError(f"{run_id}: budget {budget} > pula train {n}")
    if budget == n:
        chosen = list(range(n))
    else:
        blocks, used_weights = [], {}
        for name in GROUP_ORDER:
            if name not in active:
                continue
            x, _meta = load_group_matrix(name, pool_ids, cache_dir, seed)
            x = standardize(x)
            pca_dim = None
            if name == "dino":
                x, pca_dim = pca_dino(x, int(cfg["dino_pca_dim"]))
            w = weights[name]
            blocks.append(x.astype(np.float32) * np.float32(w))
            used_weights[name] = round(float(w), 6)
            if pca_dim is not None:
                used_weights["dino_pca_dim"] = pca_dim
        x = np.hstack(blocks)
        algo = str(cfg["cluster_algorithm"])
        if algo == "kmeans":
            labels, _centers = cluster_kmeans(
                x, budget, seed, int(cfg["kmeans_iterations"]))
        elif algo == "kcenter_greedy":
            chosen = cluster_kcenter_greedy(x, budget, return_centers=True).tolist()
        else:
            raise ValueError(f"{run_id}: nieznany cluster_algorithm '{algo}'")
        rule = str(cfg["cluster_representative"])
        if algo != "kcenter_greedy":
            chosen = pick_representatives(x, labels, budget, rule, seed)

    if len(set(chosen)) != budget:
        raise RuntimeError(f"{run_id}: wybrano {len(set(chosen))} != {budget}")
    frame_ids = sorted(pool_ids[i] for i in chosen)

    return {
        "run_id": run_id,
        "variant": run.get("variant"),
        "seed": seed,
        "budget": budget,
        "frame_ids": frame_ids,
        "selection_meta": {
            "weights": weights,
            "pca_dim": int(cfg["dino_pca_dim"]),
            "algorithm": str(cfg["cluster_algorithm"]),
            "representative": ("farthest_point_centers" if cfg["cluster_algorithm"] == "kcenter_greedy"
                               else str(cfg["cluster_representative"])),
            "groups": groups,
            "n_pool_train": n,
            "override": run.get("override"),
            "empty_cluster_rule": "first_unselected_by_frame_id_asc",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", type=Path, default=None,
                    help="queue.jsonl (domyslnie protocol.yaml paths.queue_file)")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--variants-dir", type=Path, default=None)
    ap.add_argument("--protocol", type=Path, default=None)
    ap.add_argument("--run-id", action="append", default=None,
                    help="tylko te biegi (mozna powtarzac)")
    args = ap.parse_args()

    protocol = load_protocol(args.protocol)
    paths_cfg = protocol.get("paths", {})
    feat_cfg = protocol["features"]

    queue = args.queue or resolve_under_root(
        paths_cfg.get("queue_file", "queue.jsonl"))
    manifest = args.manifest or resolve_under_root(
        paths_cfg.get("pool_manifest", "pool_manifest.jsonl"))
    cache_dir = Path(args.cache_dir or resolve_under_root(
        paths_cfg.get("feature_cache_dir", "feature_cache")))
    variants_dir = Path(args.variants_dir or resolve_under_root(
        paths_cfg.get("variants_dir", "variants")))
    variants_dir.mkdir(parents=True, exist_ok=True)

    pool = train_pool(read_manifest(manifest))
    pool_ids = [r["frame_id"] for r in pool]
    print(f"Pula train: {len(pool_ids)} klatek; cache: {cache_dir}")

    runs = []
    with open(queue, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    if args.run_id:
        runs = [r for r in runs if r["run_id"] in set(args.run_id)]

    done = skipped = 0
    for run in runs:
        run_id = run["run_id"]
        if not run.get("groups"):
            print(f"[skip] {run_id}: brak groups (baseline - baselines.py)")
            skipped += 1
            continue
        result = select_one(run, pool, pool_ids, cache_dir, feat_cfg)
        out = variants_dir / f"{run_id}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"[ok] {run_id}: {len(result['frame_ids'])}/{run['budget']} "
              f"klatek -> {out}")
        done += 1
    print(f"\nZakonczono: {done} zapisanych, {skipped} pominietych.")


if __name__ == "__main__":
    main()
