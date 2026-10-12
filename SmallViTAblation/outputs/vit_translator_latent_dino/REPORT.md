# Translator ViT do DINO — pilot

Pełny latent 600 × 768 jest bezpośrednio mapowany przez liniową regresję ridge w postaci dualnej na 384 cechy DINOv2 ViT-S/14. Nie ma SD, PCA ani uśredniania tokenów. ViT i DINO zamrożone.

240 obrazów treningowych, 75 walidacyjnych, 85 testowych; całe filmy rozłączne. Są to historyczne dane rozwojowe, więc nie jest to niezależny dowód generalizacji. Hiperparametr dobrano wyłącznie na walidacji.

```json
{
  "selection": {
    "alpha": 0.1,
    "mean_cosine": 0.8141202330589294,
    "normalized_MSE": 0.0009681236697360873,
    "retrieval_top1": 0.09333333333333334,
    "retrieval_top5": 0.29333333333333333,
    "median_rank": 14.0,
    "gallery_size": 75
  },
  "validation_trials": [
    {
      "alpha": 0.0001,
      "mean_cosine": 0.8059248328208923,
      "normalized_MSE": 0.0010108081623911858,
      "retrieval_top1": 0.08,
      "retrieval_top5": 0.32,
      "median_rank": 14.0,
      "gallery_size": 75
    },
    {
      "alpha": 0.001,
      "mean_cosine": 0.8060598373413086,
      "normalized_MSE": 0.0010101047810167074,
      "retrieval_top1": 0.08,
      "retrieval_top5": 0.32,
      "median_rank": 14.0,
      "gallery_size": 75
    },
    {
      "alpha": 0.01,
      "mean_cosine": 0.8072889447212219,
      "normalized_MSE": 0.001003703335300088,
      "retrieval_top1": 0.08,
      "retrieval_top5": 0.32,
      "median_rank": 14.0,
      "gallery_size": 75
    },
    {
      "alpha": 0.1,
      "mean_cosine": 0.8141202330589294,
      "normalized_MSE": 0.0009681236697360873,
      "retrieval_top1": 0.09333333333333334,
      "retrieval_top5": 0.29333333333333333,
      "median_rank": 14.0,
      "gallery_size": 75
    },
    {
      "alpha": 1.0,
      "mean_cosine": 0.8249943852424622,
      "normalized_MSE": 0.0009114876156672835,
      "retrieval_top1": 0.06666666666666667,
      "retrieval_top5": 0.2,
      "median_rank": 16.0,
      "gallery_size": 75
    },
    {
      "alpha": 10.0,
      "mean_cosine": 0.8251728415489197,
      "normalized_MSE": 0.0009105580975301564,
      "retrieval_top1": 0.02666666666666667,
      "retrieval_top5": 0.10666666666666667,
      "median_rank": 27.0,
      "gallery_size": 75
    }
  ],
  "test": {
    "mean_cosine": 0.8027451634407043,
    "normalized_MSE": 0.00102736905682832,
    "retrieval_top1": 0.03529411764705882,
    "retrieval_top5": 0.2,
    "median_rank": 17.0,
    "gallery_size": 85
  },
  "mean_baseline_test": {
    "mean_cosine": 0.8016747832298279,
    "normalized_MSE": 0.0010329433716833591,
    "retrieval_top1": 0.011764705882352941,
    "retrieval_top5": 0.058823529411764705,
    "median_rank": 43.0,
    "gallery_size": 85
  },
  "nearest_vit_baseline_test": {
    "mean_cosine": 0.6905799508094788,
    "normalized_MSE": 0.001611562678590417,
    "retrieval_top1": 0.03529411764705882,
    "retrieval_top5": 0.11764705882352941,
    "median_rank": 26.0,
    "gallery_size": 85
  },
  "verification": {
    "reload_passed": true,
    "full_input_shape": [
      600,
      768
    ],
    "output_dim": 384,
    "finite": true,
    "test_not_used_for_selection": true
  }
}
```

Retrieval odnosi się do galerii zdjęć z odpowiedniej odłożonej części pilota, nie do całych 35 tysięcy. Cosine nie jest procentem zgodności wartości. Sztuczny cel może leżeć poza rozkładem treningowym; jego translacja i ranking są hipotezą, a nie dowodem osiągalności ani korzyści dla uczenia detektora. Indeks FAISS musi używać dokładnie tych samych wag, CLS i preprocessingu.
