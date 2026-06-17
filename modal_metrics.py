"""
Lightweight, wandb-free live metrics publishing for Modal training runs.

Each training run writes its full metric history to a shared `modal.Dict`,
keyed by run name. A local viewer (see `metrics_viewer/`) reads the same Dict
over the authenticated Modal client SDK and renders live graphs -- nothing is
deployed to the cloud beyond the Dict itself.

This is intentionally additive: it sits alongside wandb rather than replacing
it, so a run can report to both while you evaluate the approach. If Modal (or
the Dict) is unavailable, every call here becomes a no-op and logs a warning,
so non-Modal / local runs are unaffected.
"""

import math
import threading
import time

try:
    import modal
except Exception:  # modal not installed in this environment
    modal = None

from transformers import TrainerCallback

from logger import log

# Shared between the training containers (writers) and the local viewer
# (reader). Both sides must agree on this name.
DICT_NAME = "endpointing-live-metrics"

# Keys that encode the x-axis step rather than a plottable metric.
_STEP_KEYS = ("train/global_step", "global_step", "step")

# Module-level singleton, configured once per run via `init()`.
_publisher = None


def _numeric_only(metrics):
    """Keep only finite, real-valued scalars; drop step keys, strings, bools,
    NaN/inf, and non-numeric objects (e.g. wandb.Histogram)."""
    out = {}
    for key, value in (metrics or {}).items():
        if key in _STEP_KEYS:
            continue
        if isinstance(value, (bool, str)):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out[key] = number
    return out


def _sanitize_config(config):
    out = {}
    for key, value in (config or {}).items():
        out[key] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
    return out


class MetricsPublisher:
    """Accumulates metric records keyed by training step and mirrors the full
    history into a `modal.Dict` entry on every update."""

    def __init__(self, run_name, config=None, dict_name=DICT_NAME):
        self.run_name = run_name
        self.dict_name = dict_name
        self._config = _sanitize_config(config)
        self._records = {}  # step (int) -> {metric_name: float}
        self._status = "running"
        self._last_step = 0
        self._lock = threading.Lock()
        self._warned_write_failure = False
        self._dict = None

        if modal is None:
            log.warning("modal_metrics: `modal` not importable; live metrics publishing disabled.")
            return
        try:
            self._dict = modal.Dict.from_name(dict_name, create_if_missing=True)
            log.info(f"modal_metrics: publishing live metrics to Dict '{dict_name}' as run '{run_name}'.")
        except Exception as e:
            log.warning(f"modal_metrics: could not open Dict '{dict_name}' ({e}); live metrics disabled.")

    @property
    def enabled(self):
        return self._dict is not None

    def _resolve_step(self, metrics, step):
        if step is not None:
            return int(step)
        for key in _STEP_KEYS:
            if metrics and metrics.get(key) is not None:
                return int(metrics[key])
        return self._last_step

    def publish(self, metrics, step=None):
        if not self.enabled:
            return
        resolved_step = self._resolve_step(metrics, step)
        clean = _numeric_only(metrics)
        if not clean:
            return
        with self._lock:
            record = self._records.setdefault(resolved_step, {})
            record.update(clean)
            self._last_step = max(self._last_step, resolved_step)
            self._flush_locked()

    def finish(self, status="finished"):
        if not self.enabled:
            return
        with self._lock:
            self._status = status
            self._flush_locked()

    def _flush_locked(self):
        payload = {
            "run_name": self.run_name,
            "config": self._config,
            "status": self._status,
            "last_step": self._last_step,
            "updated_at": time.time(),
            "history": [
                {"step": step, **values}
                for step, values in sorted(self._records.items())
            ],
        }
        try:
            self._dict[self.run_name] = payload
        except Exception as e:
            if not self._warned_write_failure:
                log.warning(f"modal_metrics: failed writing metrics to Dict ({e}); further failures suppressed.")
                self._warned_write_failure = True


def init(run_name, config=None):
    """Configure the module-level publisher for this run."""
    global _publisher
    _publisher = MetricsPublisher(run_name, config)
    return _publisher


def publish(metrics, step=None):
    """Publish a batch of metrics. No-op if `init()` wasn't called or publishing
    is disabled. Step is taken from the explicit arg, then a step key in
    `metrics`, then the last seen step."""
    if _publisher is not None:
        _publisher.publish(metrics, step)


def finish(status="finished"):
    if _publisher is not None:
        _publisher.finish(status)


class MetricsPublisherCallback(TrainerCallback):
    """Captures everything the HF Trainer logs (train loss, learning rate, eval
    metrics) with zero changes to existing logging logic."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            publish(logs, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        finish("finished")
