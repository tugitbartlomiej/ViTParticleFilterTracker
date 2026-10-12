"""Report completed full768 NNDMD results; no tuning or fitting here."""
from pathlib import Path
import json
import numpy as np


def main():
    root=Path(__file__).parent
    result=json.loads((root/"WYNIKI_NNDMD.json").read_text(encoding="utf-8"))
    metrics=json.loads((root/"NNDMD_METRYKI_HOLDOUT.json").read_text(encoding="utf-8"))
    selection=json.loads((root/"WYBOR_NNDMD.json").read_text(encoding="utf-8"))
    frozen=json.loads((root/"ZAMROZENIE_NNDMD.json").read_text(encoding="utf-8"))
    j=metrics["identity_J128"]["aggregate"]
    full=metrics["full768"]["aggregate"]
    static=metrics["static_EDMD_J128"]["aggregate"]
    rev=metrics["reversion_J128"]["aggregate"]
    win=selection["selected"]
    audit=json.loads((root/"NNDMD_AUDYT_OPERATORA.json").read_text(encoding="utf-8"))
    lines=["# PyKoopman NNDMD: pełne768 i końcowa tożsamośćJ128","",
           f"Wykonanie zakończone {result['completed']}. Rzeczywisty PyKoopman1.2.1 NNDMD.fit(listtrajectory), pełny encoder768→768→768, uczona macierzA768×768, decoder pełne768. J128 odczytano DOPIERO z decoded pełnych768.","",
           f"Główny wynik przyH500: znakallJ128={100*j[-1]['sign_accuracy_all128']:.3f}procent, RMSE_J128={j[-1]['rmse_state']:.6f}, RMSE_brak_zmiany={j[-1]['rmse_persistence']:.6f}, ratioRMSE={j[-1]['rmse_ratio_vs_persistence']:.6f}, skillMSE={j[-1]['mse_skill_vs_persistence']:.6f}. Cel85procent: ciągły horyzont odH10 wynosi{result['contiguous_sign85_horizon_updates']}aktualizacji; lista spełniających horyzontów: {result['any_horizons_sign85'] or 'brak'}.","",
           f"Dla pełnych768 przyH500 RMSE={full[-1]['rmse_state']:.6f}, RMSE_brak_zmiany={full[-1]['rmse_persistence']:.6f}, skillMSE={full[-1]['mse_skill_vs_persistence']:.6f}. Rekonstrukcjaorigin500: RMSEfull768={metrics['reconstruction_origin']['rmse_full768']:.6f}, RMSEJ128={metrics['reconstruction_origin']['rmse_J128']:.6f}. To osobna miara od błędu przyszłej prognozy.","",
           "## Uczony operator i strata","",
           "Alearned działa w uczonej przestrzeni768 encoderu. Ta sama liczba wymiarów nie oznacza tożsamości osi z oryginalnymi kanałami ViT. Zapisujemy uczony generatorG i A=matrix_exp(G), gdzie jeden krok odpowiada10aktualizacjom źródłowego ViT. Encoder i decoder wracają do pełnych oryginalnych768; nie ma PCA ani redukcji rzędu.","",
           "Wymiar768 nie gwarantuje injektywności encoderu ani dokładnej inwersji przez decoder. Zmierzony błąd rekonstrukcji pozostaje częścią rzeczywistej prognozy ruchu. G jest parametryzacją operatora pomiędzy checkpointami co10aktualizacji; nie jest zweryfikowanym generatorem pojedynczych aktualizacji źródłowego modelu. Historyczny tytuł PLAN_NNDMD zawiera128, ale jego późniejsze jawne korekty i faktyczny kod/trening definiują768×768.","",
           "Strata oryginalnego pakietu to wielokrokowy maskedMSE prognozy plus rekonstrukcjaorigin i prawdziwych przyszłych stanów. Jawny adapter rozszerza ją o beta razy identyczne termy decodedJ128: rnn_full+rec_full+beta*(rnn_J128+rec_J128). Rzeczywiste package sieci, generator, Trainer i Adam1e-3 zostały użyte; nie przypisujemy dodatkowej wagi identity niezmienionemu API. Wszystkie cztery składowe zapisano w CSV/logach. Łącznyloss i składowe w logach są wartościami ostatniego minibatch, a walidacyjne RMSE jest dokładną agregacją coverage i równych wag obrazów.","",
           "Do próbkowania okien użyto WeightedRandomSampler: prawdopodobieństwo coverage/sumcoverage obrazu, jednakowa masa każdego obrazu i każdego dostępnego origin. Stochastyczna realizacja wag jest poprawna w oczekiwaniu, a nie równa na każdym minibatch. Normalizacja używa jednego wspólnego skalara wyliczonego wyłącznie z fitprefixu; średnianormalizacji0. RMSE raportujemy w surowych kanałach, jako sqrt(MSE PO agregacji), nie średnia RMSE obrazów.","",
           "Dla kosztu768 liczymy matrix_exp raz na minibatch i używamy go w całym rollout. MaskęMSE wektoryzujemy, zachowując torch.where, sumę i dzielnik. Testbeta0 potwierdził zgodność loss i gradientów z oryginalnym package training_step; testbeta1 potwierdził pełnyloss plusidentity. Wolny pilot15epok zachowano osobno; główna siatka od tej samej inicjalizacji ma20epok każdego kandydata.","",
           "## Wybór bez czterech końcowych obrazów","",
           "Pierwsze24fit trenowano wyłącznie0..300, sześćfit28..33 oceniono open-loop od300 do500. Kryterium: średnia pięciu rawRMSE decodedJ128 przyH10,H20,H50,H100,H200. Najlepszą epokę spośród checkpointów co5 i kandydatów wybrano przed refit/testem. Full768RMSE raportowano dodatkowo.","",
           "| look_forward | betaidentity | Najlepsza epoka | Val mean RMSEJ128 | Val mean RMSEfull768 |",
           "|---:|---:|---:|---:|---:|"]
    for c in selection["candidates"]:
        lines.append(f"| {c['look_forward']} | {c['beta']:g} | {c['best_epoch']} | {c['validation_J128_mean_multihorizon_rmse']:.6f} | {c['best_validation_metrics']['mean_multihorizon_rmse']:.6f} |")
    baseline=audit['internal_validation_baselines']
    lines += ["",f"Na tej samej walidacji J128 i pięciu horyzontach średni RMSE braku zmiany wynosi {baseline['persistence_mean5_rmse']:.6f}, a wybranego statycznego EDMD {baseline['static_EDMD_mean5_rmse']:.6f}. Wybrany NN ma {win['validation_J128_mean_multihorizon_rmse']:.6f}. Średnia RMSE po horyzontach jest kryterium wyboru, podczas gdy każdy RMSE horyzontu powstaje po agregacji MSE po obrazach i cząsteczkach. Nie utożsamiamy jej ze sqrt(MSE uśrednionego po horyzontach)."]
    lines += ["",f"Wybrano look_forward{win['look_forward']}, beta{win['beta']:g}, epokę{win['best_epoch']}. Refit30fit na0..500 od nowej inicjalizacji z tym samymseed20261002, przez stałą liczbę wybranychepok. Zamrożono {frozen['frozen_at']}. Dopiero potem nowytest czterech starychholdoutów8..11. Forecast kodujeorigin500 RAZ; dalej tylko phi_next=A*phi i decoder, bez przyszłych obserwacji, aktualizacji modelu lub re-encodingu przewidywanych stanów.","",
              "## Horyzonty głównej tożsamościJ128","",
              "| Haktualizacji | ZnakallJ128 | RMSE NN | RMSE staticEDMD | RMSE brakzmiany | RMSE reversion | SkillNN | CosineNN |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for h in [10,20,50,100,200,300,400,500]:
        pos=h//10-1; q=j[pos]; s=static[pos]; r=rev[pos]
        lines.append(f"| {h} | {100*q['sign_accuracy_all128']:.3f}% | {q['rmse_state']:.6f} | {np.sqrt(s['mse_state']):.6f} | {q['rmse_persistence']:.6f} | {r['rmse_state']:.6f} | {q['mse_skill_vs_persistence']:.6f} | {q['cosine_motion']:.6f} |")
    lines += ["",f"PrzyH500 błądnormyruchuNN={j[-1]['norm_relative_error']:.6f}. Miary znaku obejmują wszystkie128współrzędnych ze sign(0)=0. Dodatkowo eps1e-8: znak={100*j[-1]['sign_accuracy']:.3f}procent, udział={j[-1]['active_component_mass']:.6f}; eps1e-3: znak={100*j[-1]['sign_accuracy_eps1e_3']:.3f}procent, udział={j[-1]['active_component_mass_eps1e_3']:.6f}. Cel85 dotyczy allJ128, nie progowanych miar i nie RMSE.","",
              "StaticEDMD porównano tylko wJ128, ponieważ jego stan miał oryginalne128kanałów. Nie tworzymy sztucznego pełnego768wyniku tego starego modelu. Persistence istnieje dlafull768/J128. Reversion przewiduje stały globalny ważony średni stan tokenów tool30fit na500 i nigdy nie pobiera przyszłych obserwacji.","",
              "## Diagnostyka zamrożonego operatora","",
              f"Promień spektralny A={audit['stability']['spectral_radius_A']:.6f}, norma operatorowa 2 A={audit['stability']['operator_norm2_A']:.6f}, norma A^50={audit['stability']['operator_norm2_A_power50']:.6f}. Są to miary w uczonych współrzędnych. Promień spektralny sam nie wyklucza przejściowego wzrostu ani błędu decoded prognozy.","",
              f"Liniowy decoder C ma numeryczny rząd {audit['stability']['decoder_linear_C']['rank_at_float32_relative_tolerance']} przy tolerancji float32={audit['stability']['decoder_linear_C']['tolerance']:.3g}; jego kondycja2={audit['stability']['decoder_linear_C']['condition_number2']:.6g}. Przy odwracalnym C decoded rollout można zapisać (C A C^-1)^h R(x0), gdzie R=s C E(x0/s). Sam nonlinearencoder i kwadratowy lineardecoder nie dowodzą nieliniowej sprzężoności fizycznej dynamiki.","",
              f"Reszta jednokrokowej domkniętości encoderu, RMSE względem braku zmiany w learned state: fitprefix={audit['closure']['fit30_prefix500']['learned_rmse_ratio_vs_persistence']:.6f}, końcowy holdoutfuture={audit['closure']['holdout_future_diagnostic']['learned_rmse_ratio_vs_persistence']:.6f}. Diagnostyka używa zakodowanych prawdziwych stanów dopiero po teście, bez wpływu na wybór lub prognozę. Jej jednostki to uczone, normalizowane współrzędne, a nie surowe kanały ViT. Rekonstrukcja mierzy błąd encoder/decoder; nie jest ścisłym dolnym ograniczeniem błędu przyszłej prognozy.","",
              f"J128 prognozy jest bitowo identyczny z wycinkiem zapisanej prognozy full768: {audit['decoded_J128_bitwise_equal_to_full_forecast_slice']}. Audyt nie zmieniał modelu.","",
              "## Granice wniosków","",
              "Wynik eksploracyjny: czteryholdouty były oglądane wcześniej, również w statycznymEDMD. Wszystkie34obrazy są w datasettrainViT; podział dotyczy identyfikacji dynamiki. Tylko25z30fit są w dokładnej bazie1500 źródłowego treningu(3old+22new), pozostałe5 są train poza tą bazą. J128 wybrano na500, więc wcześniejsza walidacja jest retrospektywna względem kanałów. To jedna wspólna trajektoria uczenia ViT, bez niezależnych replik. 20epok i jeden seed nie ustalają globalnego optimum neuralmodelu. Nie sprawdzano prognoz pomiędzy zapisami co10aktualizacji. Lepszyfit nie gwarantuje lepszej długiej prognozy ani85procent znaku.","",
              "## Artefakty","",
              "- PLAN_NNDMD.md: protokół i jawne korekty przed treningiem/wykonaniem.",
              "- NNDMD_PREFIX500.pt: frozen encoder, pełnygenerator768, decoder768, config i normalizacja.",
              "- NNDMD_PREFIX500_A_LEARNED_768.npy oraz GENERATOR_768.npy:768×768, learnedaxes.",
              "- WYBOR_NNDMD.json i ZAMROZENIE_NNDMD.json: wybór/chronologia/hashe.",
              "- NNDMD_PROGNOZA_HOLDOUT_FULL768.npy:50×4×600×768float32, wszystkie600cząsteczek.",
              "- NNDMD_METRYKI_HOLDOUT.json, NNDMD_HORYZONTY_J128.csv, WYNIKI_NNDMD.json: wyniki50horyzontów, perimage orazfull/J.",
              "- NNDMD_*_HISTORIA.json, nndmd_training_logs/: wszystkieepoki/kandydaty i losscomponents.",
              "- NNDMD_RUNTIME.json, NNDMD-requirements-lock.txt i smoke: wersje i weryfikacja.","",
              "- NNDMD_AUDYT_OPERATORA.json: stabilność, reszty domkniętości i kontrola decoded J128.","",
              "- NNDMD_TRAIN_EXECUTED_SNAPSHOT.py i AMENDMENT_NNDMD_SOURCE.md: dokładny hash źródła załadowanego podczas treningu, odróżniony od źródła późniejszej ewaluacji. Niezależny audyt potwierdził identyczność AST treningu/wyboru/refit; różnica dotyczy zapisu dodatkowego pliku J128 i docstringów.","",
              "```powershell", ".\\.venv-pykoopman\\Scripts\\python.exe -m pytest -q",
              ".\\.venv-pykoopman\\Scripts\\python.exe nndmd_fit.py --stage train",
              ".\\.venv-pykoopman\\Scripts\\python.exe nndmd_fit.py --stage evaluate",
              ".\\.venv-pykoopman\\Scripts\\python.exe audit_nndmd.py",
              ".\\.venv-pykoopman\\Scripts\\python.exe report_nndmd.py", "```", "",
              "Skrypt odmawia ponownego strojenia w katalogu z gotowym WYNIKI_NNDMD.json. Odtworzenie wymaga świeżego katalogu z danymi, protokołem i skryptami. RLS pozostaje odłożonym planem, bez wyników. Dawne statyczne modele i dane768/J128 źródłowych eksperymentów pozostają zachowane.","",
              "Źródła pierwotne: [oficjalny NNDMD](https://pykoopman.readthedocs.io/en/master/tutorial_koopman_nndmd_examples.html), [kodv1.2.1](https://github.com/dynamicslab/pykoopman/blob/v1.2.1/src/pykoopman/regression/_nndmd.py).",""]
    (root/"RAPORT_NNDMD_FULL768.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"Report saved: NN RMSEJ128H500={j[-1]['rmse_state']:.6f}, sign={100*j[-1]['sign_accuracy_all128']:.3f}%")


if __name__=="__main__": main()
