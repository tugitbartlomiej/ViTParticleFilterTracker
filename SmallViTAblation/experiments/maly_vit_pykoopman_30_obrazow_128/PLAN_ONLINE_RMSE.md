# Online EDMD, rekurencyjne LS i wybór macierzy przez RMSE

Zapisano2026-10-02 po zakończeniu statycznego testu, przed obliczeniami wariantu online. Wariant jest nowym, eksploracyjnym porównaniem; cztery holdouty zostały już obejrzane. Nie nadpisuje statycznych operatorów, protokołu ani wyników.

## Obliczany model i zakres

Oryginalne128kanałów J128 oraz bias. Bez PCA, redukcji rzędu, autoenkodera i obrotu reprezentacji. Własny wariant online EDMD / rekurencyjny ważony least-squares w postaci informacyjnej. Nie przypisujemy go API PyKoopman. Statyczny PyKoopman pozostaje rzeczywistym, osobno zapisanym punktem odniesienia.

Próbka u=[z,1], v=[z_next,1]. Dla dostępnych par tego samego tokenu i obrazu, coverage/sumcoverage obrazu i równe wagi obrazów: G_t=rho*G_previous+sum(w*u*u.T), C_t=rho*C_previous+sum(w*v*u.T), W_t=rho*W_previous+sumw. Aktualizacja raz na NOWY checkpoint, nie osobno na każdy token, żeby forgetting dotyczyło czasu. Wszystkie wagi w aktualizacji sumują się do1.

Macierz kolumnowa K minimalizuje sumę historycznych błędów z wagami rho^(wiek)/W_t plus alpha_t*||A-I||². Alpha_t=lambda*trace(G_t[:128,:128])/(W_t*128); stała/bias nie są karane. P=diag(1,...,1,0). K=(C_t/W_t+alpha_t*P) @ inverse(G_t/W_t+alpha_t*P), rozwiązane przez solve, bez jawnego inverse. Wszystkie129wymiary pozostają. Rho1 musi odzyskiwać ważony batch EDMD z tą samą karą; sprawdzamy zgodność z realnym PyKoopman również na rzeczywistych danych.

Chronologia0→10→20→...→500. Zanim stan t zostanie podany aktualizacji, prognoza ze stanu t-10 i poprzedniej macierzy daje prequentialRMSE. Po odczycie30fit w t dopisujemy tylko pary(t-10,t), obliczamy nową macierz i in-sampleRMSE znanej pary. Żaden przyszły stan nie jest wejściem aktualizacji. Przed pierwszą parą punkt startowy to A=I,b=0.

## Wybór kilku kandydatów

Siatka rho=[1,0.99,0.95], lambda=[0,1e-6,1e-4,1e-2],12kandydatów. Pierwsze24fit do aktualizacji0..300, ostatnie6fit(indeksy28..33) do wewnętrznej walidacji. Dla każdego kandydata zapisujemy prequentialRMSE fit i walidacji na każdym obserwowanym checkpointcie10..300; walidacja nie aktualizuje operatora.

Główne kryterium wyboru to średnia RMSE stanu sześciu obrazów walidacyjnych dla OPEN-LOOP od300 z horyzontamiH10,H20,H50,H100,H200. Każde RMSE to sqrt(MSE po coverage i równych wagach obrazów); horyzonty mają równe wagi. Dane prawdziwe310..500 służą tylko tej wewnętrznej ocenie i nie aktualizują żadnego operatora od300. Remis: rho bliżej1, potem mniejsze lambda. Nie wybieramy kandydata według czterech końcowych holdoutów. MSE/RMSE jednego kroku na train mają ten sam minimizer, ale nie gwarantują optimum błędu wielokrokowego; stąd osobny wybór przez RMSE kilku horyzontów.

Po wyborze rekurencyjny refit na30fit w porządku0..500. Zapisujemy operator po każdym checkpointcie i RMSE kolejnych znanych par, następnie zamrażamy operator na origin500. Końcowa prognoza czterech holdoutów500..1000 jest open-loop bez żadnych aktualizacji z przyszłości. Porównania: wybrany online, batchPyKoopman z TYM SAMYMlambda i prefix30fit0..500, poprzedni zamrożony PyKoopman oraz brak zmiany. Batchsame-lambda dopasowujemy realnym PyKoopman i zapisujemy oddzielnie. Rho1 online/batchsame-lambda są kontrolą zgodności, nie niezależnymi odkryciami.

Opcjonalna adaptacyjna prognoza kolejnego checkpointu z aktualizacjami po500 nie jest częścią tego wykonania. Nigdy nie oznaczamy jej jako bezpomiarowej prognozyH500.

## Metryki i smoke

Per checkpoint i horyzont: MSE, RMSE=sqrt(właściwego zagregowanegoMSE), ratioRMSE do braku zmiany, MSEskill, znak WSZYSTKICH128współrzędnych ze sign(0)=0, dodatkowo eps1e-8/1e-3 z udziałem, cosine i błąd normy. Jednostki RMSE: surowe aktywacje128kanałów GELU. Wynik85procent odnosi się do znaku, nie RMSE. Nie obiecujemy poprawy; samo rekurencyjne rozwiązanie tego samego celu z rho1 nie powinno poprawić batchoptimum.

Smoke przed obliczeniami: rekurencyjne rho1 versus niezależne ważone batchLS, rho<1 versus jawnie ważone batchLS, układ affine z biasem, checkpointpoison (zmiana przyszłych stanów nie zmienia modelu z prefixu), skutek zamiany kolejności czasu dla forgetting. Na rzeczywistych danych rho1 versus PyKoopman przy selectedlambda; tolerancja maxabs1e-8 lub jawny wynik/ograniczenie.

Wynik końcowy ma status eksploracyjny, ponieważ stareholdouty już testowano. Nie daje niezależnego potwierdzenia. Kanały wybrano na500, a wszystkie34obrazy były train ViT. Jedna trajektoria treningu i skorelowane tokeny nie stanowią niezależnych replik.
