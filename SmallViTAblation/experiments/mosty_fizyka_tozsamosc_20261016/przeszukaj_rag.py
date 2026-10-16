"""Read-only retrieval from existing Grant v2 corpus; saves evidence in ViT outputs."""
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mosty_fizyka_tozsamosc_20261016"
RAG = Path("F:/Studia/MCHTR/Grant/Projekt/Research/Wiki/my-wiki/.rag")
os.environ["RAG_KORPUS_KONFIG"] = str(RAG / "korpus.v2.json")
os.environ["HF_HOME"] = "F:/HuggingFace"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(RAG))
sys.stdout.reconfigure(encoding="utf-8")
import rag_korpus as engine

QUERIES = [
    "Optical trap active response nonconservative force stiffness matrix circulation Brownian vortex optical tweezers calibration",
    "Tozsamosc obiektu podprzestrzen niezmiennikow Koopmana V_inv etykiety pozycja kontrprzyklad D67 D115",
    "Gramian obserwowalnosci kierunki latentu wplyw na glowice wyjscie sterowalnosc balanced truncation",
    "Mori Zwanzig siec neuronowa pamiec ukryte zmienne redukcja stanu uogolnione rownanie Langevina niestacjonarny trening",
    "Huragan czastki koherentne wiry Okubo Weiss FTLE Lagrangian coherent structures deformacja tensor Cauchy Green",
    "Tozsamosc stabilna reprezentacja drift aktywna perturbacja odpowiedz impulsowa niezmienniki falsyfikacja"
]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archive = {"status": "initializing", "planned_queries": QUERIES, "queries": []}
    destination = OUT / "RAG_WYNIKI.json"
    destination.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Loading existing v2 corpus; this may be slow in embedded mode.", flush=True)
    try:
        status = engine.status()
        archive["corpus"] = status
        assert Path(status["baza"]).name == "qdrant_korpus_v2", status
        assert status["dokumenty"] == 2687 and status["chunki"] == 91489, status
        print(json.dumps({"corpus": status}, ensure_ascii=False), flush=True)
        archive["status"] = "searching"
        destination.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
        for number, query in enumerate(QUERIES, 1):
            started = time.monotonic()
            print(f"Starting query {number}/{len(QUERIES)}", flush=True)
            results = engine.search(query, top_k=5, score_mode="rerank")
            for result in results:
                result["path"] = engine.sciezka_lokalna(result["source"])
            record = {"id": number, "query": query, "mode": "rerank", "top_k": 5,
                      "seconds": time.monotonic() - started, "results": results}
            archive["queries"].append(record)
            destination.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(record, ensure_ascii=False), flush=True)
        archive["status"] = "complete"
    except Exception as exc:
        archive["status"] = "failed"
        archive["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        engine.zamknij_klienta()
        destination.write_text(json.dumps(archive, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
