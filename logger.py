import logging
import subprocess
import sys
from datetime import datetime

from torch import nn
from transformers import TrainerCallback

log = logging.getLogger("endpointing_training")
if not log.handlers:
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s \t| %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)


def log_dependencies():
    """Log all pip dependencies to console."""
    log.info("--- INSTALLED PYTHON PACKAGES ---")

    try:
        # Run pip list and capture output
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        log.info(result.stdout)

    except subprocess.CalledProcessError as e:
        log.error(f"Error running pip list: {e}")
        log.error(f"stderr: {e.stderr}")
    except Exception as e:
        log.error(f"Unexpected error logging dependencies: {e}")

    log.info("--- END DEPENDENCIES ---")


def log_model_structure(model, config):
    log.info("--- MODEL STRUCTURE AND DIMENSIONS ---")

    # Get total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    log.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Get wav2vec2 encoder layer information
    if hasattr(model, 'wav2vec2') and hasattr(model.wav2vec2, 'encoder'):
        encoder_layers = len(model.wav2vec2.encoder.layers)
        log.info(f"\nWav2Vec2 Encoder Layers: {encoder_layers}")

        # Log layer dimensions for first few layers
        log.info(f"\nLayer dimensions (first 3 layers):")
        for i in range(encoder_layers):
            layer = model.wav2vec2.encoder.layers[i]
            if hasattr(layer, 'attention'):
                attention = layer.attention
                if hasattr(attention, 'k_proj'):
                    hidden_size = attention.k_proj.in_features
                    log.info(f"  Layer {i}: hidden_size={hidden_size}, num_heads={attention.num_heads}")

    # Log transformer encoder structure
    if hasattr(model, 'transformer_encoder'):
        transformer_layers = len(model.transformer_encoder.layers)
        log.info(f"\nCustom Transformer Encoder Layers: {transformer_layers}")
        log.info(
            f"Transformer config: heads={config['transformer_heads']}, dim_feedforward={config['transformer_dim_feedforward']}")

    # Log classifier structure
    if hasattr(model, 'classifier'):
        log.info(f"\nClassifier structure:")
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, nn.Linear):
                log.info(f"  {i} Linear: {layer.in_features} -> {layer.out_features}")
            elif isinstance(layer, nn.LayerNorm):
                log.info(f"  {i} LayerNorm: normalized_shape={layer.normalized_shape}")
            else:
                log.info(f"  {type(layer).__name__} {i}")

    # Log attention pooling structure
    if hasattr(model, 'pool_attention'):
        log.info(f"\nAttention pooling structure:")
        for i, layer in enumerate(model.pool_attention):
            if isinstance(layer, nn.Linear):
                log.info(f"  {i} Linear: {layer.in_features} -> {layer.out_features}")
            else:
                log.info(f"  {i} {type(layer).__name__}")

    log.info("--- END MODEL STRUCTURE ---")


def log_dataset_statistics(split_name, dataset):
    """Log detailed statistics about each dataset split."""
    log.info(f"\n-- Dataset statistics: {split_name} --")

    # Basic statistics
    total_samples = len(dataset)
    if "labels" in dataset.features:
        labels = dataset["labels"]
        positive_samples = sum(1 for label in labels if label == 1)
        negative_samples = total_samples - positive_samples
        positive_ratio = positive_samples / total_samples * 100

        log.info(f"  Total samples: {total_samples:,}")
        log.info(f"  Positive samples (Complete): {positive_samples:,} ({positive_ratio:.2f}%)")
        log.info(f"  Negative samples (Incomplete): {negative_samples:,} ({100 - positive_ratio:.2f}%)")

        # Audio length statistics if available
        if "audio" in dataset.features:
            audio_lengths = [len(x["array"]) / 16000 for x in dataset["audio"]]  # Convert to seconds
            avg_length = sum(audio_lengths) / len(audio_lengths)
            min_length = min(audio_lengths)
            max_length = max(audio_lengths)

            log.info(f"  Audio statistics (in seconds):")
            log.info(f"    Average length: {avg_length:.2f}")
            log.info(f"    Min length: {min_length:.2f}")
            log.info(f"    Max length: {max_length:.2f}")
    else:
        log.warning(f"  (no labels!)")
        log.info(f"  Total samples: {total_samples:,}")


class ProgressLoggerCallback(TrainerCallback):
    """
    Custom callback to replace tqdm progress bars with our logging system.
    """

    def __init__(self, log_interval=50):
        self.log_interval = log_interval
        self.last_log_step = 0
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.now()
        log.info(f"Starting training with {state.max_steps} total steps")
        log.info(f"Training will run for {args.num_train_epochs} epochs")

    def on_step_end(self, args, state, control, **kwargs):
        # Log progress every log_interval steps
        if state.global_step % self.log_interval == 0 and state.global_step != self.last_log_step:
            self.last_log_step = state.global_step

            # Calculate progress percentage
            progress_pct = (state.global_step / state.max_steps) * 100

            # Calculate estimated time remaining
            if self.start_time and state.global_step > 0:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                steps_remaining = state.max_steps - state.global_step
                if elapsed_time > 0:
                    time_per_step = elapsed_time / state.global_step
                    eta_seconds = steps_remaining * time_per_step
                    eta_minutes = eta_seconds / 60

                    log.info(
                        f"Training progress: {state.global_step}/{state.max_steps} steps ({progress_pct:.1f}%) - ETA: {eta_minutes:.1f} minutes")
                else:
                    log.info(f"Training progress: {state.global_step}/{state.max_steps} steps ({progress_pct:.1f}%)")
            else:
                log.info(f"Training progress: {state.global_step}/{state.max_steps} steps ({progress_pct:.1f}%)")

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = state.epoch + 1 if state.epoch is not None else 1
        log.info(f"Starting epoch {current_epoch}/{args.num_train_epochs}")

    def on_epoch_end(self, args, state, control, **kwargs):
        current_epoch = state.epoch + 1 if state.epoch is not None else 1
        log.info(f"Completed epoch {current_epoch}/{args.num_train_epochs}")

    def on_evaluate_begin(self):
        log.info("Starting evaluation...")

    def on_evaluate_end(self, args, state, control, metrics=None):
        if metrics:
            log.info(f"Evaluation completed - Loss: {metrics.get('eval_loss', 'N/A'):.4f}, "
                     f"Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}, "
                     f"F1: {metrics.get('eval_f1', 'N/A'):.4f}")
        else:
            log.info("Evaluation completed")

    def on_save_begin(self, args, state):
        log.info(f"Saving checkpoint at step {state.global_step}...")

    def on_save_end(self):
        log.info(f"Checkpoint saved successfully")

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            total_minutes = total_time / 60
            log.info(f"Training completed successfully in {total_minutes:.1f} minutes ({total_time:.0f} seconds)")
        else:
            log.info("Training completed successfully")