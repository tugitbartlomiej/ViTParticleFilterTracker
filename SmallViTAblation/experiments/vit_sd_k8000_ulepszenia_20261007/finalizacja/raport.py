"""Render a factual final report from the transferred comparison and local test."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/vit_sd_k8000_ulepszenia_20261007'
COPY=OUT/'eden_kopia'
def read(path):return json.loads(path.read_text(encoding='utf8'))
def main():
    c=read(COPY/'wyniki/porownanie/COMPARISON.json')
    t=read(OUT/'TEST_LOKALNY.json')
    cleanup=read(OUT/'RAPORT_SPRZATANIA.json')
    visual=read(OUT/'OCENA_WIZUALNA.json')
    status=read(OUT/'TEST_LOKALNY_STATUS.json')
    winner=c['winner']; baseline=winner['baseline']; gain=1-winner['metrics']['full_relative_mse']/baseline
    text=['# SD K8000: wynik porównania i weryfikacji lokalnej', '',
          'Wszystkie cztery warianty ukończyły budżet 4000 nowych aktualizacji. Job 1823544 zakończył się COMPLETED 0:0 po 12:21:32. Nie uruchamiano ponownego treningu.', '',
          f"Najmniejszy pełny względny MSE uzyskał **A_control**, w checkpointcie po 3000 nowych aktualizacji, czyli 10500 łącznie. Błąd spadł z {baseline:.6f} do {winner['metrics']['full_relative_mse']:.6f}, o **{100*gain:.2f}%**. Żaden z trzech badanych pakietów zmian nie pokonał kontroli według tej miary.", '',
          '## Porównanie na tych samych odłożonych filmach', '',
          '| Wariant | Nowe kroki najlepszego modelu | Względny MSE VAL | Cosine VAL | Względny MSE ROI | Względny MSE TRAIN |',
          '|---|---:|---:|---:|---:|---:|']
    arms=[]
    for row in c['records']:
        b=read(COPY/'wyniki/porownanie'/row['arm']/'BEST.json')
        m=row['metrics']; tr=b['train_diagnostic']['film_mean']
        arms.append(dict(arm=row['arm'],new_steps=row['new_steps'],validation=m,train=tr))
        text.append(f"| {row['arm']} | {row['new_steps']} | {m['full_relative_mse']:.6f} | {m['cosine']:.6f} | {m['roi_relative_mse']:.6f} | {tr['full_relative_mse']:.6f} |")
    text += ['', 'A: sekwencyjne próbkowanie co 600 klatek. B: losowanie klatki w oknach 600 i mieszanie filmów. C: dane B oraz mniejszy LR z EMA. D: C z dodatkową stratą ROI. C/B mierzy pakiet LR+EMA, nie oddzielny wpływ każdego składnika.', '',
             'Walidacja: 25 filmów, 2 klatki na film, 2 seedy, 100 generacji na wariant i ocenę. Te same ustawienia odszumiania: DDIM 50 leading. Średnie kolejno po seedach, klatkach i filmach. Mała stała próbka TRAIN jest osobną diagnostyką, nie źródłem wyboru checkpointa. Pokrycie TRAIN 116/118; dwa znane filmy odrzucono z powodu niezgodnych długości MP4/CSV. Nie wyciągamy wniosku o istotności statystycznej ani o przeuczeniu z samej różnicy tych średnich.', '',
             'Pełny względny MSE = mean((z_obraz-z_cel)^2) / mean(z_cel^2), mianownik ograniczony od dołu przez 1e-6. Cosine jest osobną miarą kierunku i nie jest procentem zgodności wartości. ROI obejmuje tokeny nakładające się na ramkę narzędzia, nie wybrane 128 kanałów.', '',
             '## Warunkowanie i jakość', '',
             f"A_control: MSE właściwego warunku {winner['metrics']['full_relative_mse']:.4f}, innego filmu {winner['metrics']['wrong_condition_mse']:.4f}. Średnia indywidualnych względnych korzyści wynosi {100*winner['metrics']['relative_gain_vs_wrong_condition']:.2f}%. To dowód mierzalnej reakcji na warunek w tym porównaniu, nie gwarancja odtworzenia szczegółów.", '',
             f"Błąd ROI A wynosi {winner['metrics']['roi_relative_mse']:.4f}; IoU względem ramki źródłowej {winner['metrics']['reference_layout_iou_not_generated_ground_truth']:.4f}. To diagnostyka zachowania układu, nie trafność detekcji względem niezależnej etykiety obrazu wygenerowanego.", '',
             f"Rekonstrukcja VAE na tych samych filmach: PSNR {winner['metrics']['vae_reconstruction_psnr']:.2f} dB, SSIM {winner['metrics']['vae_reconstruction_ssim']:.4f}, względny błąd pełnego latentu {winner['metrics']['vae_reconstruction_full_relative_mse']:.4f}. Sam generator ma SSIM {winner['metrics']['ssim']:.4f} i PSNR {winner['metrics']['psnr']:.2f} dB. Nie osiągnął wcześniejszego celu względnego błędu <=0,10 ani cosine >=0,90. Dobre wyniki rekonstrukcji VAE kierują dalszą diagnostykę na generowanie i warunkowanie, ale nie stanowią pełnego dowodu przyczynowego.", '',
             visual['assessment_pl'], '',
             '## Świeży test lokalny', '',
             f"Potwierdzono SHA256 `{t['checkpoint_sha256']}`, strict load sieci i VAE oraz zgodność teachera K8000 i manifestu. Generowanie na {t['device']}, z odciążaniem pamięci przez przenoszenie nieużywanych modułów na CPU. Pełny latent 600x768; początek z losowego szumu. Dwa filmy VAL, po jednej klatce, dwa seedy. Naturalny forward ViT bez podmiany aktywacji. Wszystkie wartości skończone.", '',
             '| Film | Seed | Względny MSE | Cosine | Względny MSE ROI | MSE innego warunku | Confidence | IoU układu źródłowego |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in t['frames']:
        text.append(f"| {row['film']} | {row['seed']} | {row['full_relative_mse']:.4f} | {row['cosine']:.4f} | {row['roi_relative_mse']:.4f} | {row['wrong_condition_mse']:.4f} | {row['natural_detector_confidence']:.4f} | {row['reference_layout_iou_not_generated_ground_truth']:.4f} |")
    text += ['', 'Kontrola zamienia latenty filmów przy identycznym szumie. Ten mały test służy kontroli działania kopii i oględzinom, nie jest niezależnym testem generalizacji: obrazy pochodzą z już używanego zbioru VAL. Nie zapisano obrazów ani latentów na dysku.', '',
             'Nie wykonano generacji ze sztucznego celu: brak potwierdzonego celu utworzonego przez K8000. Nie zastąpiono go niezgodnym celem K1000/K6402. To osobne ograniczenie, nie błąd naturalnego testu.', '',
             f"Podgląd obrazów w RAM: {status['preview_url']} — działa, dopóki proces lokalnego podglądu pozostaje uruchomiony.", '',
             '## Zachowane wyniki i sprzątanie', '',
             f"Model lokalny: `{COPY/'wybrany_model/best_sd.pt'}`. Kod, logi, manifest, metryki, provenance i sumy kontrolne znajdują się w `eden_kopia`. Zweryfikowano {cleanup['local_metadata_files_verified']} małych plików.", '',
             f"Z nowej rodziny na Edenie usunięto {len(cleanup['removed'])} zbędnych checkpointów smoke i recovery, odzyskując {cleanup['bytes_freed']/1e9:.2f} GB ({cleanup['bytes_freed']/2**30:.2f} GiB). Zachowano najlepszy model każdego z czterech wariantów oraz wszystkie małe artefakty. Nie dotykano poprzednich eksperymentów, danych ani teachera. Szczegółowa lista: RAPORT_SPRZATANIA.json.", '',
             cleanup.get('scratch_note',''), '',
             '## Wniosek', '',
             'Kontynuacja bazowego przepisu pomogła liczbowo, ale badane ulepszenia nie poprawiły głównej miary względem kontroli. Najlepszy model nadal nie odtwarza wiernie narzędzia. Nie uruchomiono kolejnego treningu. Kolejny eksperyment powinien najpierw sprawdzić, czy generator potrafi wiernie zapamiętać niewielki zbiór naturalnych par, i porównać globalne cross-attention z warunkiem zachowującym jawnie siatkę przestrzenną. Są to propozycje dalszego testu, nie potwierdzone rozwiązania.', '']
    (OUT/'RAPORT_KONCOWY.md').write_text('\n'.join(text),encoding='utf8')
    (OUT/'PODSUMOWANIE_WYNIKOW.json').write_text(json.dumps(dict(winner='A_control',baseline=baseline,relative_improvement=gain,arms=arms,local_test_film_mean=t['film_mean'],visual=visual,cleanup_bytes=cleanup['bytes_freed']),indent=2,ensure_ascii=False),encoding='utf8')
    print(OUT/'RAPORT_KONCOWY.md')
if __name__=='__main__':main()
