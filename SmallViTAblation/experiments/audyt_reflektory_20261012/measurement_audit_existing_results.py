"""Read-only audit of existing result JSONs; no training or checkpoint inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/porownanie_szklo_20261010"
OUT = ROOT / "outputs/audyt_reflektory_20261012"


def read(relative):
    path = BASE / relative
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    w, sw = read("eden/wyniki/wiazki_koloidy/WIAZKI.json")
    prior = {}
    try:
        n, sn = read("film_krysztal/NADAZANIE_PULAPKI_S47_test.json")
    except FileNotFoundError:
        # This source was removed by a concurrent external task after our first
        # audit. Retain the already recorded numeric snapshot and source hash.
        prior = json.loads((OUT / "measurement_audit_existing_results.json").read_text(encoding="utf-8"))
        n = None
        sn = next(s for s in prior["sources"] if s["path"].endswith("NADAZANIE_PULAPKI_S47_test.json"))
        sn["availability_at_full_audit"] = "Removed externally after the initial read; retained source hash and summary."
    full, sf = read("film_krysztal/NADAZANIE_PULAPKI_S47.json")
    endpoint = w["cala_droga_e1_e60"]
    E = endpoint["E"]
    rows = []
    for phase, stage in w["etapy"].items():
        for lag, record in stage["lagi"].items():
            rows.append({
                "phase": phase, "lag_checkpoints": int(lag), "lag_optimizer_steps": record["krokow"],
                "sampled_checkpoint_pairs": record["n_par"],
                "reported_mean_of_W_norm_ratios": record["udzial_wiazek"],
                "complement_U_norm_ratio": 1 - record["udzial_wiazek"],
                "interaction_energy_over_observed_energy": record["E_reszta"] / record["E"],
            })
    result = {
        "sources": [sw, sn, sf],
        "scope": {"images": w["zdjec"], "checkpoints": len(w["kroki"]),
                  "first_step": w["kroki"][0], "last_step": w["kroki"][-1]},
        "endpoint_e1_e60": {
            "observed_energy": E,
            "W_energy": endpoint["E_wiazki"], "U_energy": endpoint["E_kulki"],
            "interaction_energy": endpoint["E_reszta"],
            "reported_W_ratio": endpoint["udzial_wiazek"],
            "W_energy_over_observed_energy": endpoint["E_wiazki"] / E,
            "interaction_energy_over_observed_energy": endpoint["E_reszta"] / E,
            "sum_cross_terms": E - endpoint["E_wiazki"] - endpoint["E_kulki"] - endpoint["E_reszta"],
            "interpretation": "Ratio EW/(EW+EU), not an additive fraction of actual observed energy or filtered slow drift.",
        },
        "stages": rows,
        "followup_test": {
            "test": n["test"], "settings": n["ustawienia"],
            "P1_intervals": n["P1_kontrola"]["przedzialow"],
            "actual_P1_scope_from_test_branch": [[0, 300, 10], [1000, 6000, 100]],
            "actual_P2_epochs_from_test_branch": [0, 6],
            "P1": {phase: {
                "intervals": v["przedzialow"], "optimizer_steps_per_lag": v["krok_przedzialu"],
                "C_minus1": v["F"]["C"]["-1"], "C_zero": v["F"]["C"]["0"],
                "C_plus1": v["F"]["C"]["1"], "C_null": v["F"]["zero_C_przesuniecie"],
                "forward_a1": v["F"]["regresja_pulapka_za_koloidem_a"]["1"],
                "reverse_c1": v["F"]["regresja_koloid_za_pulapka_c"]["1"],
            } for phase, v in n["P1"].items()},
        } if n is not None else prior["followup_test"],
        "followup_full": {
            "test": full["test"], "settings": full["ustawienia"],
            "P1_control": full["P1_kontrola"],
            "P1": {phase: {
                "intervals": v["przedzialow"], "optimizer_steps_per_lag": v["krok_przedzialu"],
                "C_minus1": v["F"]["C"]["-1"], "C_zero": v["F"]["C"]["0"],
                "C_plus1": v["F"]["C"]["1"], "C_null": v["F"]["zero_C_przesuniecie"],
                "forward_a1": v["F"]["regresja_pulapka_za_koloidem_a"]["1"],
                "reverse_c1": v["F"]["regresja_koloid_za_pulapka_c"]["1"],
                "C_lags_over_three_null_sd": [int(t) for t, val in v["F"]["C"].items()
                    if abs(val-v["F"]["zero_C_przesuniecie"]["sr"]) > 3*v["F"]["zero_C_przesuniecie"]["sd"]],
            } for phase, v in full["P1"].items()},
            "P2_late": {name: {phase: {
                "summary": v["podsumowanie"], "sensor_generators": v["generatory"]["s"]
            } for phase, v in group["fazy"].items() if "40–60" in phase}
                for name, group in full["P2"].items()},
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "measurement_audit_existing_results.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
    print(json.dumps(result["endpoint_e1_e60"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
