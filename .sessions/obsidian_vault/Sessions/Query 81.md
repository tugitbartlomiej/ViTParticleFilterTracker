---
date: 2025-12-13
time: 14:35
type: Mixed
topics: [Benchmark, DETR, IEEE, Pipeline, Query 81, SAM, SSH Eden, YOLO]
aliases: ["Query 81"]
---

# Query 81

> [!info] Session Info
> **Date:** 2025-12-13 14:35
> **Type:** Mixed
> **ID:** `Session_2025-12-13_143500`

## Objective

1. Rozszerzenie artykulu IEEE ACCESS o sekcje Query Specialization (Query 81)
2. Usuniecie wzmianek o zewnetrznym datasecie CaDTD
3. Dodanie opisu wlasnego narzedzia do adnotacji jako novel contribution

## Topics

[[Benchmark]] [[DETR]] [[IEEE]] [[Pipeline]] [[Query 81]] [[SAM]] [[SSH Eden]] [[YOLO]]

## Actions Taken

1. Analiza wlasnego narzedzia do adnotacji - Przeanalizowano kod w `Annotators/OpencvTrackerAnnotator/` - Przeanalizowano kod w `Annotators/Utils/` (konwertery YOLO<->COCO) - Przeanalizowano kod w `Annot

## Key Findings

- Semi-automatic tracking propagation
- Continuous annotation mode
- COCO/YOLO format support
- Augmentation pipeline

## Files Modified

- `F:\Studia\Articles\Moj\IEEE\Overleaf\DETR_IEEE\access.tex`
- `Annotators/OpencvTrackerAnnotator/opencv_annotation_tracker_movie.py`
- `Annotators/Utils/yolo_to_coco_converter.py`
- `Annotators/DetrAnnotator/coco-augmentation.py`

## Next Steps

- [ ] Skompilowac artykul w Overleaf (pdflatex && bibtex && pdflatex && pdflatex)
- [ ] Zweryfikowac czy wszystkie referencje sa rozwiazane
- [ ] Rozwazyc dodanie kodu annotation tool do supplementary materials
- [ ] Monitorowac trening DETR na Eden (job 1454332)

## Related Sessions

- [[Session]] (2025-10-20)
- [[Session]] (2025-10-20)
- [[Session]] (2025-10-21)
- [[Session]] (2025-10-22)
- [[Session]] (2025-10-29)
- [[Session]] (2025-10-31)
- [[YOLO Fix]] (Unknown)
- [[Dataset Selection]] (Unknown)
- [[DETR EL2N]] (2025-12-11)
- [[Dataset Selection]] (2025-12-11)
- [[Session]] (2025-12-11)
- [[Dataset Selection]] (2025-12-12)
- [[Dataset Selection]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Dataset Selection]] (2025-12-12)
- [[Visualization]] (2025-12-12)
- [[Session]] (2025-12-12)
- [[Query 81]] (2025-12-12)
- [[Visualization]] (2025-12-13)
- [[Visualization]] (2025-12-13)
- [[IEEE Pipeline]] (2025-12-13)
- [[Session]] (2025-12-13)
- [[Dataset Selection]] (2025-12-31)
- [[Session]] (2026-01-11)
- [[YOLO Fix]] (2026-01-11)
- [[IEEE Article]] (2026-01-12)
- [[IEEE Article]] (2026-01-12)
- [[IEEE Article]] (2026-01-12)
- [[YOLO Resume]] (2026-01-12)
- [[YOLO Resume]] (2026-01-12)
- [[Dataset Selection]] (2026-01-16)
- [[DETR EL2N]] (2026-01-18)
- [[DETR EL2N]] (2026-01-18)
- [[Dataset Selection]] (2026-01-18)
- [[YOLO Resume]] (2026-01-18)
- [[YOLO Resume]] (2026-01-19)
- [[Session]] (2026-01-20)
- [[Visualization]] (2026-01-26)
- [[Visualization]] (2026-01-26)

---

> [!tip] Navigation
> - [[Sessions Index|Back to Index]]
> - [[Mixed|All Mixed Sessions]]
