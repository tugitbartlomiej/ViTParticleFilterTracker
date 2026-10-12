# MLP pełny latent ViT → DINO

Wszystkie 460800 wartości wchodzą do pierwszej warstwy gęstej. Nie uśredniamy tokenów i nie używamy SD. Trenujemy tylko translator; checkpoint ViT i DINO pozostają zamrożone. Normalizacja wyliczona tylko na train, wybór rozmiaru MLP i kroku na val. Test na85 obrazach z innych filmów wykonany po wyborze. Historyczne dane rozwojowe nie są niezależnym testem generalizacji.

```json
{
  "chosen": {
    "hidden": 64,
    "step": 550,
    "validation": {
      "cosine": 0.7536125183105469,
      "MSE": 0.0012832680949941278,
      "top1": 0.12,
      "top5": 0.29333333333333333,
      "median_rank": 14.0,
      "gallery_size": 75
    },
    "path": "F:\\Studia\\PhD_projekt\\VIT\\ViTParticleFilterTracker\\SmallViTAblation\\outputs\\vit_translator_mlp_pelny_latent\\MLP_hidden64.pt"
  },
  "candidates": [
    {
      "hidden": 32,
      "step": 500,
      "validation": {
        "cosine": 0.764165997505188,
        "MSE": 0.0012283021351322532,
        "top1": 0.10666666666666667,
        "top5": 0.3466666666666667,
        "median_rank": 15.0,
        "gallery_size": 75
      },
      "path": "F:\\Studia\\PhD_projekt\\VIT\\ViTParticleFilterTracker\\SmallViTAblation\\outputs\\vit_translator_mlp_pelny_latent\\MLP_hidden32.pt"
    },
    {
      "hidden": 64,
      "step": 550,
      "validation": {
        "cosine": 0.7536125183105469,
        "MSE": 0.0012832680949941278,
        "top1": 0.12,
        "top5": 0.29333333333333333,
        "median_rank": 14.0,
        "gallery_size": 75
      },
      "path": "F:\\Studia\\PhD_projekt\\VIT\\ViTParticleFilterTracker\\SmallViTAblation\\outputs\\vit_translator_mlp_pelny_latent\\MLP_hidden64.pt"
    }
  ],
  "train": {
    "cosine": 0.9398066401481628,
    "MSE": 0.0003135071019642055,
    "top1": 0.9416666666666667,
    "top5": 1.0,
    "median_rank": 1.0,
    "gallery_size": 240
  },
  "test": {
    "cosine": 0.7491952776908875,
    "MSE": 0.001306274556554854,
    "top1": 0.058823529411764705,
    "top5": 0.25882352941176473,
    "median_rank": 11.0,
    "gallery_size": 85
  },
  "linear_test": {
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
  "historical_development_set": true,
  "independent_generalization": false,
  "verification": {
    "fresh_cpu_reload_passed": true,
    "max_reload_error": 1.4333054423332214e-06,
    "full_input_shape": [
      600,
      768
    ],
    "no_pooling": true,
    "output_dim": 384,
    "test_not_used_for_selection": true
  }
}
```

Cosine nie jest procentem zgodności wartości. Trafienie top1/top5 dotyczy galerii85 zdjęć. Duża różnica train/test oznacza przeuczenie. Translacja sztucznego celu to ekstrapolacja i wymaga weryfikacji znalezionych zdjęć w ViT; sama nie dowodzi korzyści dla uczenia.
