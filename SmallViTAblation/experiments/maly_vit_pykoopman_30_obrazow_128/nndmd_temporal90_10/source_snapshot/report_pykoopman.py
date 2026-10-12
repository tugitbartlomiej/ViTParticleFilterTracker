"""Post-test comparability correction and report, never refits a model."""
from pathlib import Path
import json
import csv
import numpy as np
from pykoopman_fit import forecast, sha256, write_json, now


def sign_all(delta_true,delta_pred,weights,majority):
    """All original components, including exact zeros, sign(0)=0."""
    component_weight=weights[:,None]/delta_true.shape[1]
    return {"sign_accuracy_all128":float(np.sum(component_weight*(np.sign(delta_true)==np.sign(delta_pred)))),
            "majority_sign_accuracy_all128":float(np.sum(component_weight*(np.sign(delta_true)==majority[None,:]))),
            "true_exact_zero_component_mass":float(np.sum(component_weight*(delta_true==0))),
            "pred_exact_zero_component_mass":float(np.sum(component_weight*(delta_pred==0)))}


def main():
    root=Path(__file__).parent
    result=json.loads((root/"WYNIKI.json").read_text(encoding="utf-8"))
    frozen=json.loads((root/"ZAMROZENIE_MODELU.json").read_text(encoding="utf-8"))
    assert sha256(root/"MODEL_PREFIX_500.npz")==frozen["model_sha256"]
    assert sha256(root/"PROTOKOL_MODELU.md")==frozen["protocol_sha256"]
    stored=np.load(root/"MODEL_PREFIX_500.npz")
    model={k:stored[k] for k in ["A","b"]}
    channels=stored["channels0"]
    assert model["A"].shape==(128,128) and model["b"].shape==(128,)
    for key,out in [("A","MACIERZ_KOOPMANA_128.npy"),("b","BIAS_128.npy"),
                    ("operator_affine","OPERATOR_AFFINE_129.npy"),("operator_raw","OPERATOR_PAKIETU_129.npy")]:
        np.save(root/out,stored[key])
    np.savetxt(root/"MACIERZ_KOOPMANA_128.csv",stored["A"],delimiter=",",fmt="%.17g")
    np.savetxt(root/"BIAS_128.csv",np.c_[channels,stored["b"]],delimiter=",",fmt=["%d","%.17g"],
               header="native_channel0,bias",comments="")
    latent=np.load(root/"LATENT_J128.npy",mmap_mode="r")
    masks=np.load(root/"MASKI.npz"); coverage=masks["coverage"]; tool=masks["tool"]
    majority=np.load(root/"BAZA_WIEKSZOSCI_ZNAKU.npz")["sign"]
    sets={"holdout":([8,9,10,11],"PROGNOZA_HOLDOUT.npz","METRYKI_HOLDOUT.json"),
          "fit_diagnostic":(list(range(8))+list(range(12,34)),"PROGNOZA_FIT_DIAGNOSTYKA.npz","METRYKI_FIT_DIAGNOSTYKA.json")}
    corrected={"created":now(),"amendment":"AMENDMENT_METRYKA_ZNAKU.md",
               "post_test_definition_correction":True,"no_retuning":True,
               "model_sha256":frozen["model_sha256"],
               "definition":"weighted mean of sign(delta_pred)==sign(delta_true) over all128 native components, sign(0)=0; equal image weight"}
    for group,(indices,predfile,metricsfile) in sets.items():
        saved=np.load(root/predfile)
        original=json.loads((root/metricsfile).read_text(encoding="utf-8"))
        rows=[]
        for position,image in enumerate(indices):
            origin=np.asarray(latent[50,image],dtype=np.float64)
            predicted=forecast(model,origin,50)
            assert np.array_equal(predicted.astype(np.float32),saved["prediction"][:,position]), "Replayed predictions differ from frozen test"
            ids=np.flatnonzero(tool[image]&(coverage[image]>0))
            w=coverage[image,ids].astype(np.float64); w/=w.sum()
            for h in range(1,51):
                truth=np.asarray(latent[50+h,image,ids],dtype=np.float64)-origin[ids]
                delta=predicted[h-1,ids]-origin[ids]
                values=sign_all(truth,delta,w,majority)
                old=next(r for r in original["per_image_horizon"] if r["image_index"]==image and r["horizon_updates"]==h*10)
                rows.append({"image_index":image,"horizon_updates":h*10,**values,
                             "rmse_state":float(np.sqrt(old["mse_state"])),
                             "rmse_motion":float(np.sqrt(old["mse_motion"])),
                             "rmse_persistence":float(np.sqrt(old["mse_persistence"])),
                             "rmse_ratio_vs_persistence":float(np.sqrt(old["mse_ratio_vs_persistence"])),
                             "sign_accuracy_eps1e_8":old["sign_accuracy"],
                             "included_mass_eps1e_8":old["active_component_mass"],
                             "sign_accuracy_eps1e_3":old["sign_accuracy_eps1e_3"],
                             "included_mass_eps1e_3":old["active_component_mass_eps1e_3"]})
        agg=[]
        for h in range(1,51):
            part=[r for r in rows if r["horizon_updates"]==h*10]
            entry={k:float(np.mean([r[k] for r in part])) for k in part[0] if k not in ["image_index","horizon_updates"]}
            old=original["aggregate"][h-1]
            for k in ["mse_state","mse_persistence","mse_ratio_vs_persistence","mse_skill_vs_persistence","cosine_motion","norm_relative_error"]: entry[k]=old[k]
            # Square root AFTER equal-image aggregation. Never average image RMSE.
            entry["rmse_state"]=float(np.sqrt(old["mse_state"]))
            entry["rmse_motion"]=float(np.sqrt(old["mse_motion"]))
            entry["rmse_persistence"]=float(np.sqrt(old["mse_persistence"]))
            entry["rmse_ratio_vs_persistence"]=float(np.sqrt(old["mse_ratio_vs_persistence"]))
            agg.append({"horizon_updates":h*10,"images":len(indices),**entry})
        contiguous=0
        for row in agg:
            if row["sign_accuracy_all128"]>=.85: contiguous=row["horizon_updates"]
            else: break
        corrected[group]={"image_indices":indices,"per_image_horizon":rows,"aggregate":agg,
                          "mean_sign_accuracy_all128":float(np.mean([r["sign_accuracy_all128"] for r in agg])),
                          "contiguous_sign85_horizon_updates":contiguous,
                          "any_horizons_sign85":[r["horizon_updates"] for r in agg if r["sign_accuracy_all128"]>=.85],
                          "replayed_float32_predictions_bitwise_identical":True}
        with (root/f"HORYZONTY_{group.upper()}.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(agg[0])); writer.writeheader(); writer.writerows(agg)
    write_json(root/"SIGN_ALL_COMPARABILITY.json",corrected)
    write_json(root/"METRYKI_RMSE.json",{"created":now(),"units":"raw native GELU activation channels",
               "definition":"sqrt of coverage/equal-image aggregated MSE; not average per-image RMSE",
               "holdout":corrected["holdout"],"fit_diagnostic":corrected["fit_diagnostic"]})
    hold=corrected["holdout"]
    vals=hold["aggregate"]
    exactzero=max(r["true_exact_zero_component_mass"] for r in vals)
    worst=min(vals,key=lambda r:r["sign_accuracy_all128"]); best=max(vals,key=lambda r:r["sign_accuracy_all128"])
    per500=[r for r in hold["per_image_horizon"] if r["horizon_updates"]==500]
    selected=json.loads((root/"WYBOR_MODELU.json").read_text(encoding="utf-8"))
    runtime=json.loads((root/"RUNTIME.json").read_text(encoding="utf-8"))
    descriptive=json.loads((root/"MODEL_OPISOWY_INFO.json").read_text(encoding="utf-8"))
    lines=["# Wspólna macierz Koopmana, PyKoopman: wykonany eksperyment", "",
           f"Zakończono: {result['completed']}. Fit, wewnętrzna walidacja, refit i test końcowy zostały rzeczywiście wykonane.","",
           f"**Cel85procent nie został osiągnięty.** Na czterech obrazach holdout zgodność znaku wszystkich128współrzędnych wynosi {100*vals[0]['sign_accuracy_all128']:.3f}procent dlaH10 i {100*vals[-1]['sign_accuracy_all128']:.3f}procent dlaH500. Żaden ze50sprawdzonych horyzontów nie osiąga85procent; ciągły horyzont odH10 wynosi0.","",
           f"PrzyH500 MSE_model={vals[-1]['mse_state']:.8f}, MSE_brak_zmiany={vals[-1]['mse_persistence']:.8f}, ratio={vals[-1]['mse_ratio_vs_persistence']:.6f}, skill={vals[-1]['mse_skill_vs_persistence']:.6f}. Model ma zatem o{100*(vals[-1]['mse_ratio_vs_persistence']-1):.2f}procent większy błąd niż brak zmiany. Średnia zgodność znaku po50horyzontach: {100*hold['mean_sign_accuracy_all128']:.3f}procent. To wynik negatywny tego ustalonego modelu affine, nie dowód niemożliwości innych modeli.","",
           f"RMSE przyH500: model={vals[-1]['rmse_state']:.8f}, brak zmiany={vals[-1]['rmse_persistence']:.8f}, ratio={vals[-1]['rmse_ratio_vs_persistence']:.6f}. Jednostki to surowe aktywacje oryginalnych kanałów GELU. RMSE jest pierwiastkiem z odpowiedniego MSE po agregacji coverage i równych wag obrazów, nie średnią RMSE obrazów. RMSE ruchu jest identyczne z RMSE stanu, ponieważ oba błędy różnią się tylko odjęciem tego samego origin. Cel85procent odnosi się do znaku, nie RMSE. Minimalizowanie MSE i RMSE na tych samych danych ma ten sam minimizer.","",
           "## Operator i rzeczywiste dopasowanie","",
           "A128×128 i b128 są wspólne dla wszystkich tokenów narzędzia wszystkich30obrazów. Zapisana orientacja: z_next=A@z+b dla kolumn; dla wierszy z_next=z@A.T+b. Operator obejmuje10aktualizacji treningu. [z,1] daje operator129×129. Kanały z KANALY_J128.npz pozostają oryginalne i w ustalonej kolejności.","",
           "Dopasowanie wykonał pykoopman.Koopman(Identity,EDMD(svd_rank=-1,tlsq_rank=0)). Nie użyto własnego solvera zamiast pakietu. Jawnie ważone [z,1] obu stron zachowują bias. Pełna wewnętrzna SVD nie odrzuca żadnego wymiaru; zapisana macierz w oryginalnej bazie to U@state_matrix@U.T.","",
           f"Prefix0..500 obejmuje1500przejść obraz-czas i{result['fit_token_pairs']}par tokenowych. Wagi coverage sumują się do1 na obrazie, obrazy mają równe wagi. Lambda={result['selected_lambda']}, alpha={result['operator_diagnostics']['alpha']:.10g}; dopisano128sztucznych par odpowiadających karze alpha*||A-I||². Bias bez tej kary. Suma wag rzeczywistych par1.","",
           f"Pełny zachowany wymiar129; błąd ostatniego wiersza stałej {result['operator_diagnostics']['constant_row_maxabs_error']:.3e}, zgodność orientacji z package.predict {result['operator_diagnostics']['package_predict_orientation_maxabs']:.3e}, promień spektralnyA={result['operator_diagnostics']['spectral_radius_A']:.6f}, condition design={result['operator_diagnostics']['condition_number_design']:.3f}. Stałą w prognozie affine ustawiamy jawnie1; zapisujemy także surowy operator pakietu.","",
           "## Wybór i zamrożenie","",
           "Pierwsze24fit: identyfikacja0..300; ostatnie6fit(indeksy28..33): open-loop300..500. Wybór według średniego ilorazu MSE po20horyzontach10..200. Cztery holdouty8..11 nie brały udziału w wyborze. Wewnętrzna walidacja również nie była dobra:","",
           "| lambda | Średni MSE ratio walidacji | Znak, eps1e-8 |","|---:|---:|---:|"]
    for c in selected["candidates"]: lines.append(f"| {c['lambda']:g} | {c['validation_mean_mse_ratio']:.6f} | {100*c['validation_mean_sign_accuracy']:.3f}% |")
    lines += ["",f"Refit wszystkich30fit zakończył się zamrożeniem {frozen['frozen_at']}; test zakończył się {result['completed']}. Prognoza500..1000 używa tylko origin500 i zamrożonego operatora. Nie ma podawania przyszłych obserwacji. Wybór modelu, zamrożenie i test są osobnymi plikami i etapami CLI.","",
              "## Holdout, horyzonty10..500","",
              "| Horyzont aktualizacji | Znak all128 | RMSE model | RMSE brak zmiany | Skill MSE | Cosine ruchu | Błąd normy |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for h in [10,50,100,200,300,400,500]:
        q=vals[h//10-1]
        lines.append(f"| {h} | {100*q['sign_accuracy_all128']:.3f}% | {q['rmse_state']:.6f} | {q['rmse_persistence']:.6f} | {q['mse_skill_vs_persistence']:.6f} | {q['cosine_motion']:.6f} | {q['norm_relative_error']:.6f} |")
    lines += ["",f"Najniższy znak={100*worst['sign_accuracy_all128']:.3f}procent przyH{worst['horizon_updates']}; najwyższy={100*best['sign_accuracy_all128']:.3f}procent przyH{best['horizon_updates']}. DlaH500 osobne obrazy:","",
              "| Indeks obrazu | Znak all128 | Znak większościowy fit |","|---:|---:|---:|"]
    for q in per500: lines.append(f"| {q['image_index']} | {100*q['sign_accuracy_all128']:.3f}% | {100*q['majority_sign_accuracy_all128']:.3f}% |")
    lines += ["", "## Jawna korekta definicji znaku po teście","",
              "Oryginalny przedtestowy protokół używał eps1e-8 i eps1e-3. Po otwarciu testu audyt przypomniał historyczny cel85 dla wszystkich128składowych z sign(0)=0. AMENDMENT_METRYKA_ZNAKU.md dokumentuje dodanie all128 jako korekty porównywalności po teście. WYNIKI.json i METRYKI_HOLDOUT.json zachowują pierwotną definicję; SIGN_ALL_COMPARABILITY.json ma wszystkie trzy. Nie dopasowano ponownie modelu, nie zmieniono wyboru lambda. Odtworzone prognozy w float32 są bitowo identyczne z wcześniej zapisanymi.","",
              f"Maksymalny udział dokładnie zerowych prawdziwych składowych={exactzero:.3g}. PrzyH500 udział składowych w eps1e-8={vals[-1]['included_mass_eps1e_8']:.8f}, w eps1e-3={vals[-1]['included_mass_eps1e_3']:.8f}; all128={100*vals[-1]['sign_accuracy_all128']:.3f}procent, eps1e-8={100*vals[-1]['sign_accuracy_eps1e_8']:.3f}procent, eps1e-3={100*vals[-1]['sign_accuracy_eps1e_3']:.3f}procent.","",
              "## Macierz opisowa0..1000","",
              f"Po końcowym teście osobno dopasowano MODEL_OPISOWY_0_1000.npz na wszystkich30fit i wszystkich100parach czasowych: {descriptive['pairs']['image_time_transitions']}przejść obraz-czas i{descriptive['pairs']['real_token_pairs']}par tokenowych. Lambda odziedziczona z wcześniejszej walidacji. To macierz opisowa, która zna przyszłość względem origin500; NIE użyto jej w żadnej przedstawionej prognozie.","",
              "## Granice wniosków","",
              "Wszystkie34obrazy były w train ViT. Holdout dotyczy wyłącznie identyfikacji dynamiki i cztery obrazy były oglądane w dawnych badaniach. 22nowefit i3z8starychfit należą do dokładnej bazy1500;5starychfit pochodzi z train poza bazą. J128 wybrano na500, więc wewnętrzna walidacja300..500 jest retrospektywna względem wyboru kanałów. Jeden bieg uczenia daje skorelowane cząsteczki i pomiary, bez niezależnych replik treningu. Dysponujemy101checkpointami co10aktualizacji; nie oceniamy pojedynczych aktualizacji pomiędzy zapisami.","",
              "## Artefakty i odtworzenie","",
              "- MACIERZ_KOOPMANA_128.npy/CSV: wspólneA w oryginalnych kanałach.",
              "- BIAS_128.npy/CSV: bias w kolejności channels0.",
              "- MODEL_PREFIX_500.npz: A,b,operator_raw,operator_affine,basis,internal_operator,eigenvalues,singular_values,channels0.",
              "- PROGNOZA_HOLDOUT.npz:50×4×600×128 float32; prognozy wszystkich tokenów, metryki wyłącznie narzędzia.",
              "- SIGN_ALL_COMPARABILITY.json, METRYKI_HOLDOUT.json, HORYZONTY_HOLDOUT.csv: wszystkie horyzonty i obrazy.",
              "- PROGNOZA_FIT_DIAGNOSTYKA.npz i METRYKI_FIT_DIAGNOSTYKA.json: diagnostyka fit, oddzielona od holdout.",
              "- PROTOKOL_MODELU.md,WYBOR_MODELU.json,ZAMROZENIE_MODELU.json: chronologia bez retuningu.",
              "- MODEL_OPISOWY_0_1000.npz i MODEL_OPISOWY_INFO.json: dodatkowy operator opisowy.",
              "- RUNTIME.json, requirements-lock.txt, RUNTIME_PIP_CHECK.log, SMOKE.log: środowisko i sprawdzenia.","",
              f"PyKoopman{runtime['versions']['pykoopman']}, PyDMD{runtime['versions']['pydmd']}, NumPy{runtime['versions']['numpy']}, SciPy{runtime['versions']['scipy']}, scikit-learn{runtime['versions']['scikit-learn']}, Python3.11.8. Pełna izolowana .venv-pykoopman; brak zmian globalnego F:\\Python. Przypięto setuptools80.9, ponieważ setuptools84 nie zawiera pkg_resources wymaganej przy imporcie pakietu. uv pip check:206pakietów zgodnych. Testy syntetyczne przeszły, istnieją ostrzeżenia deprecacji zależności pkg_resources/lightning.","",
              "```powershell", ".\\.venv-pykoopman\\Scripts\\python.exe -m pytest -q",
              ".\\.venv-pykoopman\\Scripts\\python.exe pykoopman_fit.py --stage fit",
              ".\\.venv-pykoopman\\Scripts\\python.exe pykoopman_fit.py --stage evaluate",
              ".\\.venv-pykoopman\\Scripts\\python.exe pykoopman_fit.py --stage descriptive",
              ".\\.venv-pykoopman\\Scripts\\python.exe report_pykoopman.py", "```", "",
              "Fit/evaluate celowo odmawiają ponownego strojenia w katalogu z gotowym WYNIKI.json; odtworzenie wymaga świeżego katalogu z kopiami danych, protokołu i skryptów. To zabezpiecza przed dopasowywaniem do otwartego holdoutu.","",
              "Źródła pierwotne: [EDMD v1.2.1](https://raw.githubusercontent.com/dynamicslab/pykoopman/v1.2.1/src/pykoopman/regression/_edmd.py), [oficjalny przykład par](https://pykoopman.readthedocs.io/en/master/tutorial_koopman_edmd_with_rbf.html), [release](https://github.com/dynamicslab/pykoopman/releases/tag/v1.2.1).",""]
    (root/"RAPORT_PYKOOPMAN.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"H10_all128":vals[0]["sign_accuracy_all128"],"H500_all128":vals[-1]["sign_accuracy_all128"],
                      "goal85_horizon":hold["contiguous_sign85_horizon_updates"],"zero_mass_max":exactzero},indent=2))


if __name__=="__main__": main()
