# Session: CIFAR-10 (DeiT-Tiny i MałyViT) na Edenie, pobrania i porządki na Edenie

**Created:** 2026-10-13 16:38
**Type:** Training / SSH
**Status:** Active (4 biegi MałegoViT wznowione na Edenie)

## Quick Summary
Drugi model i zbiór do analizy laserów, pułapek i koloidów: DeiT-Tiny i MałyViT użytkownika na CIFAR-10, plus biegi kontrolne
przed publikacją (ziarno 48, bez weight decay, losowe etykiety). Na Edenie zabrakło miejsca, więc pobieram lokalnie gotowe katalogi,
sprawdzam MD5 i dopiero wtedy usuwam je z Edena. Rejestr pobrań i usunięć: `SESSION_SUMMARY.md`, sekcja „Pobrania i usunięcia”.

## Key Results
- DeiT-Tiny CIFAR-10 (60 epok): 0,750 z obcięciem, 0,753 bez obcięcia.
- MałyViT CIFAR-10: ~0,85 po 50–80 tys. krokach (wyraźnie lepiej niż DeiT); losowe etykiety 0,10 (zgodnie z oczekiwaniem).
- CIFAR-10 z HuggingFace zgodny z oryginałem z Toronto obraz po obrazie (50 000 + 10 000).

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
