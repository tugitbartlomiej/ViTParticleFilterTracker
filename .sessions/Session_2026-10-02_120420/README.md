# Session: Session_2026-10-02_120420

**Created:** 2026-10-02 12:04:20 +02:00, Europe/Warsaw  
**Type:** Mixed — analiza, literatura, benchmark, transfer danych  
**Status:** Active — zapis wykonany; nowy eksperyment PyKoopman trwa

**Najnowszy stan, 13:25:** trening i test NNDMD zakończone. Rzeczywiste wymiary to **wejście 768, learned state 768, decoder 768, A 768×768**. Wcześniejsza propozycja learned state 128 została zastąpiona. Cel 85 procent znaku nie został osiągnięty. Zweryfikowany wynik i audyt są w [końcowym uzupełnieniu](UPDATE_NNDMD_FINAL_1325.md); poniższe akapity zachowują wcześniejsze etapy zapisu.

**Aktualizacja 14:53:** użytkownik zlecił nowy model rozszerzony o historię zmian z768 i charakterystyki aktualizacji wag/AdamW, na dokładnie 1500 obrazach, z treningiem na Edenie. Lokalne wcześniejsze przebiegi mają być zatrzymane. Nowy wynik nie jest jeszcze znany; stan weryfikacji i granice przyczynowości opisuje [uzupełnienie rozszerzonego NNDMD](UPDATE_EXPANDED_EDEN_1453.md).

## Quick Summary

Podsumowanie dostępnej historii sesji o cząsteczkach ViT: od definicji tokenów i researchu Koopmana na wagach, przez pobranie danych, 400 par oryginał–LaMa, identyfikację kierunków i 128 oryginalnych kanałów, po prognozy Langevina/Koopmana/Kalmana do 500 aktualizacji. Jest to rzeczowy zapis ustaleń i artefaktów, nie dosłowna transkrypcja rozmowy.

Potwierdzone rozpoznanie ROI na 128 kanałach osiągnęło 92,83 procent punktowej trafności testowej. Oddzielny benchmark dynamiki nie osiągnął celu 85 procent znaku. Nowe dane PyKoopman obejmują 30 obrazów fit i dokładnie 4 zachowane holdouty, czyli 34 obrazy; raport ekstrakcji jest gotowy, ale wyników nowego dopasowania jeszcze nie potwierdzono.

**Późniejsza aktualizacja:** statyczny EDMD został obliczony, a użytkownik zlecił osobną gałąź NNDMD. Po kolejnej korekcie jej wejściem mają być pełne 768 współrzędnych cząsteczki, z rekonstrukcją 768D; J128 służy dodatkowej ocenie tożsamości. Wyuczony latent 128D jest propozycją konstrukcji, nie ostatecznie potwierdzonym wymiarem. Stan i chronologia są w postscriptum poniżej.

## Files

- [SESSION_SUMMARY.md](SESSION_SUMMARY.md) — pełne podsumowanie, dowody, ograniczenia i dalsze kroki.
- [UPDATE_NNDMD.md](UPDATE_NNDMD.md) — późniejsze zlecenie uczonego encodera/decodera; aktualizacja stanu EDMD i dokumentu pipeline.
- [UPDATE_NNDMD_FINAL_1325.md](UPDATE_NNDMD_FINAL_1325.md) — faktyczne pełne768, ukończony NNDMD i niezależnie zweryfikowany ujemny wynik prognoz.
- [Reguły zapisu sesji](../SESSION_RULES.md).
- [UPDATE_EXPANDED_EDEN_1453.md](UPDATE_EXPANDED_EDEN_1453.md) — nowa autoryzacja Eden, pełne768 z historią, manifest1500, sparse AdamW i audyt przed uruchomieniem.
- [Mechanizm indeksowania](../tools/README.md).
- [Status indeksowania i kontroli zapisu](INDEX_STATUS.json).

## Related Sessions

- [Poprzedni zapis animacji pipeline ViT](../Session_2026-10-01_042605/SESSION_SUMMARY.md).
- [Wcześniejsza sesja o siłach i kierunkach cząsteczek](../Session_2026-09-29_145251_Maly_ViT_Petla_Sily_Tozsamosci_i_Kierunki_Czastek/SESSION_SUMMARY.md).
