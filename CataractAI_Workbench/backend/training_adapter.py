"""Adapter wrapping BackgroundFinetuned/main_train.py for GUI integration."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Ensure project root is on sys.path so BackgroundFinetuned modules are importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from CataractAI_Workbench.app.core.project_paths import TRAINING_SETTINGS


class TrainingAdapter:
    """High-level adapter between the GUI and ConfigurableTrainer."""

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = Path(config_path) if config_path else TRAINING_SETTINGS

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def load_config(self) -> dict:
        """Load and return the training JSON config."""
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_path}")
        with open(self._config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_config(self, config: dict, path: Optional[str] = None) -> None:
        """Save *config* dict to JSON (defaults to the loaded path)."""
        dest = Path(path) if path else self._config_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def get_default_config(self) -> dict:
        """Return a sensible default config matching TrainingSettings.json schema."""
        return {
            "training_configuration": {
                "description": "DETR training configuration",
                "version": "2.0",
            },
            "model_settings": {
                "checkpoint_path": "",
                "base_model": "facebook/detr-resnet-50",
                "num_labels": 1,
                "ignore_mismatched_sizes": True,
            },
            "dataset_paths": {
                "mixed_dataset_dir": "",
                "images_dir": "all_images",
                "annotations_file": "annotations/mixed_annotations.json",
                "auto_use_dino_subsets": True,
                "dino_subset_suffix": ".dino",
            },
            "training_parameters": {
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": 2e-6,
                "weight_decay": 1e-5,
                "warmup_epochs": 1,
                "gradient_clip": 1.0,
                "images_per_epoch": None,
            },
            "optimization": {
                "optimizer": "AdamW",
                "scheduler": {"type": "StepLR", "step_size": 2, "gamma": 0.7},
                "use_amp": True,
            },
            "early_stopping": {
                "enabled": True,
                "patience": 6,
                "min_delta": 0.0005,
                "monitor": "loss",
            },
            "checkpointing": {
                "save_best": True,
                "save_every_n_epochs": 3,
                "save_final": True,
                "checkpoint_dir": str(_PROJECT_ROOT / "BackgroundFinetuned" / "output"),
            },
            "device_settings": {
                "device": "auto",
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": True,
            },
            "logging": {
                "log_level": "INFO",
                "save_training_plots": True,
                "save_training_history": True,
                "progress_bar": False,
                "log_dir": str(
                    _PROJECT_ROOT / "BackgroundFinetuned" / "Logs" / "training"
                ),
            },
            "augmentation": {
                "enabled": True,
                "random_seed": 42,
                "reproducible": True,
            },
            "validation": {
                "validation_split": 0.0,
                "validation_frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def run_training(
        self,
        config_overrides: Optional[dict] = None,
        epoch_callback: Optional[Callable[[dict], None]] = None,
        cancel_flag: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Run the training loop via ConfigurableTrainer.

        Parameters
        ----------
        config_overrides : dict, optional
            Keys to override in *training_parameters* (epochs, batch_size,
            learning_rate, etc.).
        epoch_callback : callable, optional
            Called after every epoch with a dict containing at least
            ``{epoch, total_epochs, loss, lr, tool_batches, bg_batches}``.
        cancel_flag : callable, optional
            A callable returning True when training should stop early.

        Returns
        -------
        dict  with keys: final_model, history, best_loss, epochs_ran
        """
        # Lazy import -- avoids loading torch at module level
        from BackgroundFinetuned.main_train import ConfigurableTrainer

        # Write a temporary config if we need to apply overrides
        config = self.load_config()

        if config_overrides:
            tp = config.setdefault("training_parameters", {})
            for key, value in config_overrides.items():
                if key in ("epochs", "batch_size", "images_per_epoch"):
                    tp[key] = int(value)
                elif key == "learning_rate":
                    tp["learning_rate"] = float(value)
                elif key == "weight_decay":
                    tp["weight_decay"] = float(value)
                elif key == "warmup_epochs":
                    tp["warmup_epochs"] = int(value)
                elif key == "gradient_clip":
                    tp["gradient_clip"] = float(value)

        # Disable tqdm progress bar for GUI mode
        config.setdefault("logging", {})["progress_bar"] = False

        # Save the (possibly modified) config to a temporary file so the
        # trainer picks it up without us needing to patch internals.
        tmp_cfg = (
            Path(config["checkpointing"].get("checkpoint_dir", "."))
            / "_workbench_training_config.json"
        )
        tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        trainer = ConfigurableTrainer(
            config_path=str(tmp_cfg), test_mode=False, verbose=False
        )

        # --- Monkey-patch the trainer to inject our callback + cancel --------
        original_train_epoch = trainer.train_epoch

        history: Dict[str, list] = {
            "losses": [],
            "learning_rates": [],
            "tool_batches": [],
            "background_batches": [],
        }
        total_epochs = config["training_parameters"]["epochs"]

        def _patched_run() -> dict:
            """Re-implement the training loop so we can intercept each epoch."""
            if not trainer.validate_dataset():
                raise RuntimeError("Dataset validation failed")

            import random
            import numpy as np
            import torch
            import torch.optim as optim

            aug = config.get("augmentation", {})
            if aug.get("reproducible", False):
                seed = aug.get("random_seed", 42)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            model, proc = trainer.load_model_and_processor()
            dl, _ = trainer.create_dataloader(proc)

            tp = config["training_parameters"]
            opt_cfg = config["optimization"]

            # Build optimizer
            optimizer_cls = {
                "AdamW": optim.AdamW,
                "Adam": optim.Adam,
                "SGD": optim.SGD,
            }.get(opt_cfg.get("optimizer", "AdamW"), optim.AdamW)

            optimizer = optimizer_cls(
                model.parameters(),
                lr=tp["learning_rate"],
                weight_decay=tp.get("weight_decay", 1e-5),
            )

            sched_conf = opt_cfg.get("scheduler", {})
            sched_type = sched_conf.get("type", "StepLR")
            if sched_type == "CosineAnnealingLR":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=total_epochs
                )
            else:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=sched_conf.get("step_size", 2),
                    gamma=sched_conf.get("gamma", 0.7),
                )

            scaler = (
                torch.cuda.amp.GradScaler()
                if opt_cfg.get("use_amp", True)
                else None
            )

            best_loss = float("inf")
            patience_counter = 0
            es = config.get("early_stopping", {})
            final_model_path = None
            epochs_ran = 0

            for ep in range(1, total_epochs + 1):
                # Check cancellation
                if cancel_flag and cancel_flag():
                    break

                avg_loss, tb, bb = original_train_epoch(
                    model, dl, optimizer, scaler, ep
                )
                current_lr = optimizer.param_groups[0]["lr"]

                history["losses"].append(avg_loss)
                history["learning_rates"].append(current_lr)
                history["tool_batches"].append(tb)
                history["background_batches"].append(bb)

                scheduler.step()
                epochs_ran = ep

                # Early stopping bookkeeping
                improved = avg_loss < best_loss - es.get("min_delta", 0.0005)
                if improved:
                    best_loss = avg_loss
                    patience_counter = 0
                    ckpt = config.get("checkpointing", {})
                    if ckpt.get("save_best", True):
                        trainer.save_model(model, ep, avg_loss, "_best")
                else:
                    patience_counter += 1

                ckpt = config.get("checkpointing", {})
                save_n = ckpt.get("save_every_n_epochs", 0)
                if save_n and ep % save_n == 0:
                    trainer.save_model(model, ep, avg_loss, f"_epoch{ep}")

                # Epoch callback
                if epoch_callback:
                    epoch_callback(
                        {
                            "epoch": ep,
                            "total_epochs": total_epochs,
                            "loss": avg_loss,
                            "lr": current_lr,
                            "tool_batches": tb,
                            "bg_batches": bb,
                        }
                    )

                if es.get("enabled", True) and patience_counter >= es.get(
                    "patience", 6
                ):
                    break

            # Save final model
            ckpt = config.get("checkpointing", {})
            if ckpt.get("save_final", True):
                final_model_path = str(
                    trainer.save_model(
                        model, epochs_ran, history["losses"][-1], "_final"
                    )
                )

            return {
                "final_model": final_model_path,
                "history": history,
                "best_loss": best_loss,
                "epochs_ran": epochs_ran,
            }

        result = _patched_run()

        # Clean up temp config
        try:
            tmp_cfg.unlink(missing_ok=True)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def list_checkpoints(self, checkpoint_dir: str) -> List[dict]:
        """Scan *checkpoint_dir* for ``.pth`` files and return metadata."""
        cdir = Path(checkpoint_dir)
        if not cdir.exists():
            return []

        results: List[dict] = []
        for p in sorted(cdir.glob("*.pth"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = p.stat()
            # Try to parse epoch / loss from filename
            epoch = None
            loss = None
            name = p.stem
            for part in name.split("_"):
                if part.startswith("epoch"):
                    try:
                        epoch = int(part.replace("epoch", ""))
                    except ValueError:
                        pass

            results.append(
                {
                    "path": str(p),
                    "filename": p.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "epoch": epoch,
                    "loss": loss,
                }
            )
        return results
