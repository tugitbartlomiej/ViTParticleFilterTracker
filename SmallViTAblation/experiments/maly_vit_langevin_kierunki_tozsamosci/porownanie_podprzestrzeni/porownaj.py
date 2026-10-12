"""Czy zamrożona podprzestrzeń k500 zachowuje orientację do k1000?

Duże kąty główne obalają interpretację, że oba checkpointy wyznaczają tę
samą podprzestrzeń. Ten pomiar jest retrospektywny i nie wybiera cech ani
hiperparametrów prognozy. Znak lub kolejność kolumn SVD nie są porównywane.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('early')
    parser.add_argument('late')
    parser.add_argument('--early-key', default='U')
    parser.add_argument('--late-key', default='U')
    args = parser.parse_args()
    early_path, late_path = Path(args.early), Path(args.late)
    with np.load(early_path, allow_pickle=False) as a:
        early = a[args.early_key].astype(np.float64)
    with np.load(late_path, allow_pickle=False) as a:
        late = a[args.late_key].astype(np.float64)
    assert early.shape == late.shape == (768, 8)
    orth_errors = [float(np.linalg.norm(u.T @ u - np.eye(8))) for u in (early, late)]
    assert max(orth_errors) < 1e-5
    q0 = np.linalg.qr(early)[0]
    q1 = np.linalg.qr(late)[0]
    cosines = np.clip(np.linalg.svd(q0.T @ q1, compute_uv=False), 0, 1)
    angles = np.degrees(np.arccos(cosines))
    overlap = float(np.mean(cosines**2))
    distance = float(np.sqrt(max(0, 16 - 2*np.sum(cosines**2))))
    result = dict(
        status='complete', role='retrospective_only_not_forecast_input',
        checkpoints=[500, 1000], early_path=str(early_path.resolve()),
        late_path=str(late_path.resolve()), early_key=args.early_key, late_key=args.late_key,
        early_sha256=sha(early_path), late_sha256=sha(late_path),
        dimensions=768, rank=8, orthogonality_errors=orth_errors,
        principal_angles_degrees=angles.tolist(),
        mean_squared_cosine=overlap, projector_frobenius_distance=distance,
        interpretation='Angles concern subspaces, not matched individual SVD columns. '
        'Overlap is geometric, not a percentage of semantic identity or forecast accuracy.')
    (HERE/'POROWNANIE.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(np.arange(1, 9), angles, color='#3c7199')
    ax.set(xlabel='Numer kąta głównego (nie numer osi latentu)',
           ylabel='Kąt [stopnie]', ylim=(0, 90), xticks=np.arange(1, 9),
           title='Zmiana podprzestrzeni: checkpoint 500 wobec 1000')
    ax.grid(axis='y', alpha=.2)
    for i, angle in enumerate(angles, 1):
        ax.text(i, angle+1, f'{angle:.1f}°', ha='center', fontsize=9)
    fig.savefig(HERE/'KATY_PODPRZESTRZENI.png', dpi=160)
    plt.close(fig)
    (HERE/'WYNIKI.md').write_text(
        '# Czy zmieniła się wyznaczona podprzestrzeń tożsamości?\n\n'
        'Porównanie opisowe ośmiu kierunków z checkpointów 500 i 1000. '
        'Nie jest wejściem ani kryterium wyboru modelu prognostycznego.\n\n'
        'Kąty główne: '+', '.join(f'{x:.2f}°' for x in angles)+'.\n\n'
        f'Średnia kwadratów cosinusów wynosi {overlap:.6f}; norma Frobeniusa różnicy projektorów: {distance:.6f}. '
        'Zgodne podprzestrzenie dawałyby kąty 0°, wartość 1 i odległość 0. '
        'To wskaźniki geometrii, nie procent zachowanej tożsamości narzędzia.\n\n'
        'Makrowektory bazowe e_j pozostają stałymi osiami współrzędnych. '
        'Przesuwać mogą się cząsteczki, a wyznaczane z danych kierunki tożsamości mogą tworzyć inną podprzestrzeń. '
        'W modelu Langevina opisanym obok zamrożono kierunki z checkpointu 500. '
        'Ta decyzja oddziela ruch cząsteczek od zmiany układu pomiarowego.\n\n'
        '![Kąty główne podprzestrzeni](KATY_PODPRZESTRZENI.png)\n\n'
        'Metoda: QR usuwa jedynie błąd numeryczny ortonormalności; wartości osobliwe Q500ᵀ Q1000 '
        'są cosinusami kątów głównych. Wynik nie zależy od kolejności lub znaku kolumn obu baz. '
        'Pełna zgodność wyniku po rotacji baz stanowi kontrolę implementacji.\n', encoding='utf-8')
    rng = np.random.default_rng(20261002)
    rot0, rot1 = [np.linalg.qr(rng.normal(size=(8, 8)))[0] for _ in range(2)]
    alt = np.linalg.svd((q0 @ rot0).T @ (q1 @ rot1), compute_uv=False)
    assert np.allclose(cosines, np.clip(alt, 0, 1), atol=1e-12)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
