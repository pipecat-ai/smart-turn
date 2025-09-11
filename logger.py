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

    # Get Whisper encoder layer information
    if hasattr(model, 'encoder'):
        encoder_layers = len(model.encoder.layers)
        log.info(f"Whisper Encoder Layers: {encoder_layers}")

        # Get first layer info for dimensions
        if encoder_layers > 0:
            first_layer = model.encoder.layers[0]
            if hasattr(first_layer, 'self_attn'):
                self_attn = first_layer.self_attn
                embed_dim = self_attn.embed_dim
                num_heads = self_attn.num_heads
                log.info(f"  Encoder embed_dim: {embed_dim}, num_heads: {num_heads}")

    # Log classifier structure
    if hasattr(model, 'classifier'):
        log.info(f"\nClassifier structure:")
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, nn.Linear):
                log.info(f"  {i} Linear: {layer.in_features} -> {layer.out_features}")
            elif isinstance(layer, nn.LayerNorm):
                log.info(f"  {i} LayerNorm: normalized_shape={layer.normalized_shape}")
            else:
                log.info(f"  {i} {type(layer).__name__}")

    # Log attention pooling structure
    if hasattr(model, 'pool_attention'):
        log.info(f"\nAttention pooling structure:")
        for i, layer in enumerate(model.pool_attention):
            if isinstance(layer, nn.Linear):
                log.info(f"  {i} Linear: {layer.in_features} -> {layer.out_features}")
            else:
                log.info(f"  {i} {type(layer).__name__}")

    # Log Whisper configuration details
    if hasattr(model, 'config'):
        whisper_config = model.config
        log.info(f"\nWhisper Configuration:")
        log.info(f"  Model dimension (d_model): {getattr(whisper_config, 'd_model', 'N/A')}")
        log.info(f"  Encoder layers: {getattr(whisper_config, 'encoder_layers', 'N/A')}")
        log.info(f"  Decoder layers: {getattr(whisper_config, 'decoder_layers', 'N/A')}")
        log.info(f"  Encoder attention heads: {getattr(whisper_config, 'encoder_attention_heads', 'N/A')}")
        log.info(f"  Max source positions: {getattr(whisper_config, 'max_source_positions', 'N/A')}")

    log.info("--- END MODEL STRUCTURE ---")


def log_dataset_statistics(split_name, dataset):
    """Log detailed statistics about each dataset split - handles both HF datasets and OnDemandWhisperDataset."""
    log.info(f"\n-- Dataset statistics: {split_name} --")

    # Basic statistics
    total_samples = len(dataset)
    log.info(f"  Total samples: {total_samples:,}")

    # Check if it's our custom dataset
    if hasattr(dataset, 'dataset'):
        # It's OnDemandWhisperDataset - access the underlying HF dataset
        underlying_dataset = dataset.dataset
        log.info(f"  Dataset type: OnDemandWhisperDataset (with on-demand preprocessing)")

        # Check if underlying dataset has the endpoint_bool column
        if hasattr(underlying_dataset, 'column_names') and 'endpoint_bool' in underlying_dataset.column_names:
            endpoint_labels = underlying_dataset['endpoint_bool']
            positive_samples = sum(1 for label in endpoint_labels if label)
            negative_samples = total_samples - positive_samples
            positive_ratio = positive_samples / total_samples * 100

            log.info(f"  Positive samples (Complete): {positive_samples:,} ({positive_ratio:.2f}%)")
            log.info(f"  Negative samples (Incomplete): {negative_samples:,} ({100 - positive_ratio:.2f}%)")

        # Log language distribution if available
        if hasattr(underlying_dataset, 'column_names') and 'language' in underlying_dataset.column_names:
            languages = underlying_dataset['language']
            from collections import Counter
            lang_counts = Counter(languages)
            log.info(f"  Language distribution: {dict(lang_counts)}")

        # Log other metadata columns
        if hasattr(underlying_dataset, 'column_names'):
            other_columns = [col for col in underlying_dataset.column_names
                             if col not in ['audio', 'endpoint_bool', 'language']]
            if other_columns:
                log.info(f"  Other available columns: {other_columns}")

        # Try to get a sample to show feature dimensions
        try:
            sample = dataset[0]  # This will trigger preprocessing
            if 'input_features' in sample:
                feature_shape = sample['input_features'].shape
                log.info(f"  Processed feature shape: {feature_shape}")
                # Estimate duration from feature shape (each frame ~10ms for Whisper)
                if len(feature_shape) >= 2:
                    n_frames = feature_shape[-1]
                    estimated_duration = n_frames * 0.01  # 10ms per frame for Whisper
                    log.info(f"  Estimated max duration: {estimated_duration:.1f} seconds")
        except Exception as e:
            log.info(f"  Could not analyze processed features: {e}")

    elif hasattr(dataset, 'features'):
        # It's a regular HF dataset
        log.info(f"  Dataset type: HuggingFace Dataset")

        if "labels" in dataset.features:
            labels = dataset["labels"]
            positive_samples = sum(1 for label in labels if label == 1)
            negative_samples = total_samples - positive_samples
            positive_ratio = positive_samples / total_samples * 100

            log.info(f"  Positive samples (Complete): {positive_samples:,} ({positive_ratio:.2f}%)")
            log.info(f"  Negative samples (Incomplete): {negative_samples:,} ({100 - positive_ratio:.2f}%)")

            # For Whisper, we work with input_features instead of raw audio
            if "input_features" in dataset.features:
                try:
                    sample_features = dataset[0]["input_features"]
                    if hasattr(sample_features, 'shape') and len(sample_features.shape) >= 2:
                        n_frames = sample_features.shape[-1]
                        estimated_duration = n_frames * 0.01  # 10ms per frame for Whisper
                        log.info(f"  Feature statistics:")
                        log.info(f"    Sample feature shape: {sample_features.shape}")
                        log.info(f"    Estimated duration (first sample): {estimated_duration:.2f} seconds")
                    else:
                        log.info(f"  Feature statistics:")
                        log.info(f"    Features available but shape analysis not possible")
                except Exception as e:
                    log.info(f"  Feature statistics:")
                    log.info(f"    Could not analyze feature dimensions: {e}")
        else:
            log.warning(f"  (no labels found in features)")
    else:
        # Unknown dataset type
        log.info(f"  Dataset type: {type(dataset).__name__}")
        log.warning(f"  Could not analyze dataset structure")


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