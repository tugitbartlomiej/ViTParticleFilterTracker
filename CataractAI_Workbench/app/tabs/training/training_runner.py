"""QThread worker that drives a DETR training run via TrainingAdapter."""

from ...core.worker_base import PipelineWorker
from ...core.signal_bus import get_signal_bus


class TrainingRunner(PipelineWorker):
    """Run DETR training in a background thread.

    Signals emitted
    ----------------
    progress(int, str)   -- percent complete + message
    log_message(str)     -- per-epoch log line
    stage_changed(str)   -- "training"
    result_ready(dict)   -- final results when done
    error_occurred(str, str) -- on failure

    The global signal bus also receives ``epoch_completed``,
    ``training_started``, and ``training_finished``.
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config

    def run_task(self) -> dict:
        # Import here to keep module import lightweight
        from CataractAI_Workbench.backend.training_adapter import TrainingAdapter

        bus = get_signal_bus()
        model_name = self._config.get("model_settings", {}).get(
            "base_model", "DETR"
        )

        self.stage_changed.emit("training")
        bus.training_started.emit(model_name)
        self.emit_log(f"Starting training: {model_name}")

        adapter = TrainingAdapter()

        # Save the current config so the adapter picks it up
        adapter.save_config(self._config)

        total_epochs = self._config.get("training_parameters", {}).get("epochs", 5)

        def _epoch_cb(data: dict) -> None:
            ep = data.get("epoch", 0)
            loss = data.get("loss", 0.0)
            lr = data.get("lr", 0.0)
            tb = data.get("tool_batches", 0)
            bb = data.get("bg_batches", 0)

            pct = int(ep / total_epochs * 100)
            msg = (
                f"Epoch {ep}/{total_epochs}  "
                f"loss={loss:.4f}  lr={lr:.2e}  "
                f"tool={tb}  bg={bb}"
            )
            self.emit_progress(pct, msg)
            self.emit_log(msg)

            # Forward to global bus
            bus.epoch_completed.emit(data)

        try:
            result = adapter.run_training(
                epoch_callback=_epoch_cb,
                cancel_flag=lambda: self._cancelled,
            )
        except Exception as exc:
            bus.training_finished.emit(model_name, {"error": str(exc)})
            raise

        bus.training_finished.emit(model_name, result)
        self.emit_log(
            f"Training finished. Best loss: {result.get('best_loss', '?'):.4f}  "
            f"Epochs: {result.get('epochs_ran', '?')}"
        )
        return result
