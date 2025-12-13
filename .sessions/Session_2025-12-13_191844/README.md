# Session: TestDatasetGenerator Pipeline

**Created:** 2025-12-13 19:18:44
**Type:** Tools
**Status:** Completed

## Quick Summary
Stworzono kompletny pipeline do generowania test datasetu z filmów wideo dla modeli YOLO/DETR. Pipeline obejmuje: ekstrakcję losowych klatek z wideo, wykrywanie duplikatów względem datasetu treningowego, automatyczną adnotację DETR, manualny review w OpenCV oraz konwersję COCO→YOLO.

## Key Results
- 7 skryptów w folderze `TestDatasetGenerator/`
- 2448 klatek wyekstrahowanych z 27 filmów
- 589 obrazów zaadnotowanych i zweryfikowanych (do obrazu 1040)
- Pipeline: DETR auto-annotate → OpenCV review → COCO to YOLO export

## Files
- `SESSION_SUMMARY.md` - Pełna dokumentacja
