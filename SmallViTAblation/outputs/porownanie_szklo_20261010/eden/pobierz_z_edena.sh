#!/bin/bash
# Pobiera wyniki z Edena (2026-10-10_szklo_koloidalne) do lokalnych katalogów, z kontrolą SHA-256 z manifestów.
#   bash pobierz_z_edena.sh szklo            # MODEL_SZKLA_EDEN.json + log + osie czasu symulacji (stala, stygnace)
#   bash pobierz_z_edena.sh pomiar <folder>  # oś czasu pomiaru wielu zdjęć (np. pomiar_pelny_3000tr_1000val, ~6,5 GB)
# Nic nie jest zapisywane ani kasowane po stronie Edena. Osie trafiają do outputs/ruch_czasteczek_gui, JSON-y do eden/wyniki.
set -euo pipefail
TRYB=${1:?szklo | pomiar <folder>}
R=/home2/faculty/bpiotrowski/smallvit_runs/2026-10-10_szklo_koloidalne/wyniki
HERE="$(cd "$(dirname "$0")" && pwd)"
OSIE="$(cd "$HERE/../../ruch_czasteczek_gui" && pwd)"
mkdir -p "$HERE/wyniki"

pobierz_os() {                                     # $1 = zdalny folder osi, $2 = lokalna nazwa
  local zd="$1" lok="$OSIE/$2"
  mkdir -p "$lok"
  scp -q -o BatchMode=yes "eden-cluster:$zd/manifest.json" "$lok/manifest.json"
  for extra in metrics.npy initial_pca.npz images.json OCENY_POMIAR.json pca32_podzbior.npz; do
    scp -q -o BatchMode=yes "eden-cluster:$zd/$extra" "$lok/$extra" 2>/dev/null || true
  done
  # klatki jednym strumieniem tar przez ssh (MSYS psuje listy ścieżek /home2/... w argumentach scp)
  ssh -o BatchMode=yes eden-cluster "cd '$zd' && tar cf - step_*.q16" | tar xf - -C "$lok"
  "F:/Python/3.13/python.exe" - "$lok" <<'EOF'
import json, sys, hashlib, os
lok = sys.argv[1]
m = json.load(open(os.path.join(lok, 'manifest.json'), encoding='utf-8'))
brak, zle = [], []
for s, f in m['files'].items():
    p = os.path.join(lok, f['name'])
    if not os.path.isfile(p):
        brak.append(f['name']); continue
    if hashlib.sha256(open(p, 'rb').read()).hexdigest() != f['sha256']:
        zle.append(f['name'])
print(f'{os.path.basename(lok)}: {len(m["files"])} klatek, brak {len(brak)}, SHA niezgodne: {zle if zle else "brak"}', flush=True)
sys.exit(1 if (brak or zle) else 0)
EOF
}

case "$TRYB" in
  szklo)
    for t in szybko_stala szybko_stygnace stala stygnace; do
      scp -q -o BatchMode=yes "eden-cluster:$R/$t/MODEL_SZKLA_EDEN.json" "$HERE/wyniki/MODEL_SZKLA_EDEN_$t.json" || { echo "brak wyników $t"; continue; }
      scp -q -o BatchMode=yes "eden-cluster:$R/$t/log.txt" "$HERE/wyniki/log_$t.txt" || true
      for zd in $(ssh -o BatchMode=yes eden-cluster "ls -d $R/$t/osie/*" 2>/dev/null); do
        pobierz_os "$zd" "20261010T_szklo_eden_$(basename "$zd")"
      done
    done;;
  pomiar)
    F=${2:?nazwa folderu pomiaru na Edenie, np. pomiar_pelny_3000tr_1000val}
    pobierz_os "$R/$F" "20261010T_natural_eden_${F#pomiar_}";;
  *) echo "tryb: szklo | pomiar <folder>"; exit 2;;
esac
echo "gotowe"
