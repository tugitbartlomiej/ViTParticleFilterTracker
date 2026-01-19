# Session Summary: Scientific Justification Dataset Selection

## Metadata
- **Date:** 2026-01-18
- **Time:** 14:35:00
- **Status:** Completed
- **Type:** Analysis / Research

## Objective
Znaleźć naukowe uzasadnienie dla metodologii selekcji datasetu używanej w projekcie AdvancedDatasetSelection, w tym dla:
- EL2N scores jako miary trudności próbek
- DINO embeddings do klastrowania
- Fourier features do analizy częstotliwościowej
- Weighted coreset selection

## Context
Sesja jest kontynuacją prac nad Advanced Dataset Selection. Wcześniej:
- Naprawiono problem z PyTorch GPU (wersja CPU → CUDA 12.4)
- Uruchomiono pipeline selekcji 20k obrazów z 90k
- Przeprowadzono analizę feature importance (Random Forest)
- Wyniki pokazały: DINO 48.4%, Fourier 46.3%, SAM 5.3%

Użytkownik zapytał o naukowe uzasadnienie dla arbitralnych wyborów (PCA dim, wagi cech).

## Actions Taken

1. **Web Search: EL2N Score**
   - Znaleziono "Deep Learning on a Data Diet" (Paul et al., NeurIPS 2021)
   - Kluczowy paper wprowadzający EL2N jako metrykę trudności próbek
   - 50% pruning CIFAR-10 bez spadku accuracy

2. **Web Search: Coreset Selection**
   - "Blind Coreset Selection" (ICLR 2025)
   - "InfoMax Coreset Selection" (ICLR 2025)
   - Survey: "A Coreset Selection of Coreset Selection Literature" (2025)

3. **Web Search: Fourier Features**
   - "Fourier Features Let Networks Learn High Frequency Functions" (NeurIPS 2020)
   - "Fourier Spectrum Discrepancies in Deep Network Generated Images" (NeurIPS 2020)
   - Uzasadnienie dla analizy częstotliwościowej w selekcji

4. **Web Search: DINO Features**
   - DINO (Meta AI, 2021)
   - DINOv2 (2023), DINOv3 (2025)
   - Self-supervised representations dla semantic coverage

5. **Zapisanie do pamięci Sereny**
   - Utworzono `dataset_selection_scientific_justification.md`
   - Pełne referencje, cytaty, rekomendowane wagi, wzorcowy cytat IEEE

## Results

### Key Findings

#### EL2N Score (NeurIPS 2021)
> "Using EL2N scores calculated a few epochs into training, we can prune half of the CIFAR10 training set while slightly improving test accuracy."

#### Coreset Selection (ICLR 2025)
> "A well-selected coreset balances the original dataset's geometric coverage while preserving its structural diversity."

#### Fourier Features (NeurIPS 2020)
> "Standard MLPs have rapid frequency falloff, preventing them from representing high-frequency content in natural images."

#### DINO (ICCV 2021)
> "Self-supervised ViT features contain explicit information about semantic segmentation."

### Rekomendowane Wagi (na podstawie feature importance + literatura)
| Cecha | Feature Importance | Rekomendowana Waga |
|-------|-------------------|-------------------|
| DINO | 48.4% | 48% |
| Fourier | 46.3% | 46% |
| SAM | 5.3% | 5% |
| EL2N | (target) | 1% |

### Issues Encountered
- Brak bezpośrednich problemów technicznych
- Wyszukiwanie wymagało kilku iteracji dla różnych aspektów metodologii

## Files Generated/Modified

### Created:
- `F:\..\.sessions\Session_2026-01-18_143500_Scientific_Justification_Dataset_Selection\`
  - `README.md`
  - `SESSION_SUMMARY.md`

### Serena Memory:
- `dataset_selection_scientific_justification.md` - pełna dokumentacja z referencjami

## Commands Used

```python
# Web Search queries
WebSearch("EL2N score Deep Learning on a Data Diet dataset pruning")
WebSearch("coreset selection deep learning training efficiency feature importance 2024 2025")
WebSearch("Fourier frequency features image classification CNN dataset diversity")
WebSearch("DINO features self-supervised learning dataset pruning 2024 2025")

# Serena Memory
mcp__plugin_serena_serena__write_memory("dataset_selection_scientific_justification.md", content)
```

## Next Steps
- [ ] Opcjonalnie: Przepuścić pipeline z nowymi wagami (DINO 48%, Fourier 46%, SAM 5%)
- [ ] Użyć przygotowanego cytatu w artykule IEEE
- [ ] Rozważyć dodanie dodatkowych referencji do Related Work

## References

### Główne Publikacje:
1. M. Paul, S. Ganguli, G.K. Dziugaite, "Deep Learning on a Data Diet: Finding Important Examples Early in Training," NeurIPS 2021. [arXiv:2107.07075](https://arxiv.org/abs/2107.07075)

2. "Blind Coreset Selection: Efficient Pruning for Unlabeled Data," ICLR 2025. [OpenReview](https://openreview.net/forum?id=pGINxZWjK4)

3. M. Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains," NeurIPS 2020. [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)

4. M. Caron et al., "Emerging Properties in Self-Supervised Vision Transformers," ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

5. "A Coreset Selection of Coreset Selection Literature," 2025. [arXiv:2505.17799](https://arxiv.org/abs/2505.17799)

---

*Sesja zapisana: 2026-01-18 14:35*
*Projekt: ViTParticleFilterTracker*
