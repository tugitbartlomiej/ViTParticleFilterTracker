# Wynik paczki32

Autentyczny stan AdamW k1000; sonda29 TR_0154/1926, bez ingerencji w latent. Prognoza stałej początkowej siły zapisana przed treningiem. Jedna epoka puli32 to jeden krok, nie epoka pełnego zbioru.

{
  "initial": {
    "mse768": 14.320196091522003,
    "cosine768": 0.12702268135879197,
    "mse128": 85.92117654913206,
    "cosine128": 0.10673829500773405
  },
  "first_step": {
    "selected32": {
      "full_with_BN_skill_percent": 94.69990311142617,
      "fixed_old_BN_skill_percent": 99.91768523106728,
      "predicted": {
        "mse768": 14.33246308892699,
        "cosine768": 0.12351389664975902,
        "mse128": 85.99147423956842,
        "cosine128": 0.09723307736300818
      },
      "actual_full_BN": {
        "mse768": 14.333868493671773,
        "cosine768": 0.12308525443714402,
        "mse128": 85.9992743397523,
        "cosine128": 0.0960719726829432
      },
      "actual_fixed_BN": {
        "mse768": 14.332508249272266,
        "cosine768": 0.12351543873554627,
        "mse128": 85.99170591491483,
        "cosine128": 0.09723831475477669
      }
    },
    "random32": {
      "full_with_BN_skill_percent": 99.35231893466533,
      "fixed_old_BN_skill_percent": 99.82303513461919,
      "predicted": {
        "mse768": 14.323975324533425,
        "cosine768": 0.12619590880799544,
        "mse128": 85.94183625717966,
        "cosine128": 0.10454289448968143
      },
      "actual_full_BN": {
        "mse768": 14.324150468484003,
        "cosine768": 0.12613155257177255,
        "mse128": 85.94294302395707,
        "cosine128": 0.10434244624798489
      },
      "actual_fixed_BN": {
        "mse768": 14.323992545267389,
        "cosine768": 0.12617751883570189,
        "mse128": 85.94192305456427,
        "cosine128": 0.10450284973159978
      }
    }
  },
  "forecast_hits": {
    "selected32": {
      "original": {
        "128": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        },
        "768": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        }
      },
      "historically_fitted": {
        "128": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        },
        "768": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        }
      }
    },
    "random32": {
      "original": {
        "128": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        },
        "768": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        }
      },
      "historically_fitted": {
        "128": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        },
        "768": {
          "0.1": null,
          "0.25": null,
          "0.5": null
        }
      }
    }
  },
  "actual_hits": {
    "selected32": {
      "128": {
        "0.1": null,
        "0.25": null,
        "0.5": null
      },
      "768": {
        "0.1": null,
        "0.25": null,
        "0.5": null
      }
    },
    "random32": {
      "128": {
        "0.1": null,
        "0.25": null,
        "0.5": null
      },
      "768": {
        "0.1": null,
        "0.25": null,
        "0.5": null
      }
    }
  }
}

null oznacza brak trafienia w horyzoncie50. Nie dowodzi to globalnej niemożliwości osiągnięcia celu po zmianie wag. Współczynniki historyczne dopasowano do średnich narzędzia, transfer na wszystkie600 tokenów oceniono pierwszym krokiem. Test pojedynczej sondy, bez wnioskowania o generalizacji.
