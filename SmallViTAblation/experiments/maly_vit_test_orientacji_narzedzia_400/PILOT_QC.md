# Pilot LaMa: kontrola6 par

2026-10-01 Europe/Warsaw. Obejrzano PILOT_LAMA_6.jpg: oryginał, lokalnaLaMa i rzeczywista maskaedycji,6 różnych orientacji. Zamaskowany trzon i końcówka narzędzia usunięte bez oczywistej długiej pozostałości w tych6przykładach; syntetyczne wypełnienie zmienia lokalną teksturę/odbicia. Pozostałe instrumenty poza maską pozostają zgodnie z zasadącompositing. Nie certyfikuje to braku artefaktów na400obrazach ani odzyskania prawdziwego tła. Pierwszy przykład ma jasne drobne struktury przy brzegu tęczówki; nie interpretować ich automatycznie jako odtworzonej anatomii.

Runtime istniejący ONNXruntime1.19.2CPU,4wątki,batch1. Pierwsza inferencja2,12s, następne1,70–1,78s. IOprzetestowane: image float32[1,3,512,512], mask[1,1,512,512], outputfloat[1,3,512,512],0..255. RzeczywisteIObatchdynamic, cardopisuje1. Wszystkie6 nativeoutcomposites mają0zmienionych pikseli poza masceedycji. ParametryINPAINT_CONFIG.json zamrożone dla całych400. Wystartowano400inferencji z pominięciem6 już poprawnie zapisanych par, bez zmianyparametrów.

Nie jest to forward badanego ViT/detektora; jedyny model lokalnej generacji toLaMa. GPU nie wykorzystano. ModelcacheHF_HOME=F:/HuggingFace, revision iSHA wMODEL.json. Nie użytoimagegen ani usług generacji.

## Uzupełnienie — 2026-10-01 17:24:29 Europe/Warsaw

Dodatkowa plansza 12 par daje łącznie 17 unikalnych ocenianych par. W natywnej TR_0022_S1_P03/00003266 znaleziono ciemną linię trzonu. Pilot radius10/radius15 ograniczył ślad; zastosowano radius10 i zachowano poprzedni obraz. Reszta400 nie otrzymała automatycznie większej maski. Wszystkie finalne pary mają dokładnie zachowane piksele poza maską; brak gwarancji pełnego usunięcia artefaktów w nieoglądanej reszcie. Kandydaci są dodatkowymi artefaktami diagnostycznymi poza800obrazami zbioru.
