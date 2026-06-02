import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import onnx
import torch
import wandb
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, quant_pre_process, \
    QuantFormat, CalibrationMethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, WhisperPreTrainedModel, WhisperConfig
# noinspection PyProtectedMember
from transformers.models.whisper.modeling_whisper import WhisperEncoder
# noinspection PyProtectedMember
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from .audio_utils import truncate_audio_to_last_n_seconds
from .benchmark import benchmark
from datasets import load_dataset, concatenate_datasets, load_from_disk
from .logger import log, log_model_structure, log_dataset_statistics, log_dependencies, log_device_info, ProgressLoggerCallback

CONFIG = {
    "base_model_name": "openai/whisper-tiny",

    "datasets_training": ["pipecat-ai/smart-turn-data-v3.2-train"],
    "datasets_test": ["pipecat-ai/smart-turn-data-v3.2-test"],

    "learning_rate": 5e-5,
    "num_epochs": 4,
    "train_batch_size": 384,
    "eval_batch_size": 128,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,

    "onnx_opset_version": 20,
    "calibration_dataset_size": 1024,

    "freeze_encoder_layers": 0,
    "load_best_model_at_end": False,
}


def ensure_torchcodec_available():
    """
    HF `datasets` audio decoding for this project requires `torchcodec` at runtime.
    Fail fast with a helpful error message if it's missing.
    """
    try:
        import torchcodec  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError("Missing required dependency 'torchcodec'") from e


class SmartTurnV3Model(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        config.max_source_positions = 400
        self.encoder = WhisperEncoder(config)

        # Use the encoder's hidden size
        hidden_size = config.d_model

        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Initialize classifier weights
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        # Initialize attention pooling weights
        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_features, labels=None):
        """
        Forward pass using Whisper encoder only

        Args:
            input_features: Log-mel spectrogram features [batch_size, n_mels, n_frames] - now (batch_size, 80, 800)
            labels: Binary labels for endpointing (1 = complete, 0 = incomplete)
        """
        # Use only the encoder part of Whisper
        encoder_outputs = self.encoder(input_features=input_features)

        hidden_states = encoder_outputs.last_hidden_state

        attention_weights = self.pool_attention(hidden_states)
        attention_weights = softmax(attention_weights, dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)

        logits = self.classifier(pooled)

        if labels is not None:
            # Calculate positive sample weight based on batch statistics
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            probs = torch.sigmoid(logits.detach())
            return {"loss": loss, "logits": probs}

        probs = torch.sigmoid(logits)
        return {"logits": probs}


class CalibrationDataset:
    """Calibration dataset for ONNX quantization with stratified sampling (early-stop)."""

    def __init__(self, dataset, feature_extractor, max_samples):

        log.info("Building calibration dataset...")

        self.feature_extractor = feature_extractor

        ds = dataset.dataset.shuffle(seed=42)
        n = min(max_samples, len(ds))
        subset = ds.select(range(n))
        self.dataset = subset

        label_view = subset.select_columns(["endpoint_bool"])
        labels = label_view["endpoint_bool"]
        pos = sum(1 for v in labels if v)
        neg = len(labels) - pos

        log.info(f"Calibration dataset: {n} samples (positives={pos}, negatives={neg})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio_array = sample["audio"]["array"]
        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )
        return inputs.input_features.squeeze(0).numpy()


class ONNXCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset):
        self.calibration_dataset = calibration_dataset
        self.iterator = iter(range(len(calibration_dataset)))

    def get_next(self):
        try:
            idx = next(self.iterator)
            input_data = self.calibration_dataset[idx]  # shape (80, 800)
            input_data = np.expand_dims(input_data, axis=0)  # shape (1, 80, 800)
            input_data = input_data.astype(np.float32, copy=False)
            return {"input_features": input_data}
        except StopIteration:
            return None


def export_to_onnx_fp32(model, output_path, config):
    """Export model to ONNX FP32 format"""
    try:
        log.info("Exporting model to ONNX FP32...")

        class ONNXExportWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_features):
                out = self.inner(input_features)
                logits = out["logits"] if isinstance(out, dict) else out
                # Output (batch_size, 1) shape (2D) - standard format for single-output models
                batch_size = logits.shape[0]
                return logits.reshape(batch_size, 1)

        export_model = ONNXExportWrapper(model).eval().cpu()

        example_input_b1 = torch.randn(1, 80, 800)
        example_input_b2 = torch.randn(2, 80, 800)

        # Test with both batch size 1 and 2 to ensure consistent output shapes
        with torch.no_grad():
            test_output_1 = export_model(example_input_b1)
            test_output_2 = export_model(example_input_b2)
            assert test_output_1.shape == (1, 1), f"Expected (1, 1), got {test_output_1.shape}"
            assert test_output_2.shape == (2, 1), f"Expected (2, 1), got {test_output_2.shape}"

        torch.onnx.export(
            model=export_model,
            args=(example_input_b2,),
            f=output_path,
            export_params=True,
            opset_version=config["onnx_opset_version"],
            do_constant_folding=False,
            input_names=['input_features'],
            output_names=['logits'],
            dynamic_axes={
                'input_features': {0: 'batch_size'},
                'logits': {0: 'batch_size'},
            },
            verbose=False,
            dynamo=False,
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        log.info(f"FP32 ONNX model saved to {output_path}")

        # Verify the exported model works with batch sizes 1 and 2 and outputs consistent shapes
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)

        example_input_1_np = example_input_b1.numpy().astype(np.float32)
        outputs_1 = session.run(None, {'input_features': example_input_1_np})
        assert outputs_1[0].shape == (1, 1), f"Expected (1, 1), got {outputs_1[0].shape}"

        example_input_2_np = example_input_b2.numpy().astype(np.float32)
        outputs_2 = session.run(None, {'input_features': example_input_2_np})
        assert outputs_2[0].shape == (2, 1), f"Expected (2, 1), got {outputs_2[0].shape}"

        log.info("ONNX model verification successful - consistent output shapes for both batch sizes")

        return output_path

    except Exception:
        log.exception("Failed to export to ONNX")
        return None


def quantize_onnx_model(
        onnx_fp32_path: str,
        training_dataset,
        feature_extractor,
        exports_path,
        calibration_dataset_size: int):
    """Quantize ONNX model using static quantization"""

    log.info("Invoking quant_pre_process...")

    pre_path = os.path.join(exports_path, "model_pre.onnx")
    quant_pre_process(
        onnx_fp32_path,
        pre_path,
        skip_optimization=False,  # let it fold/clean
        skip_symbolic_shape=True,
        verbose=1
    )

    log.info(f"Invoking quantize_static for calibration dataset size: {calibration_dataset_size} ...")

    output_path = os.path.join(exports_path, f"model_int8_static_calib{calibration_dataset_size}.onnx")

    log.info("Building calibration dataset...")

    calibration_dataset = CalibrationDataset(
        training_dataset,
        feature_extractor,
        max_samples=calibration_dataset_size,
    )

    log.info("Invoking quantize_static...")

    quantize_static(
        model_input=pre_path,
        model_output=output_path,
        calibration_data_reader=ONNXCalibrationDataReader(calibration_dataset),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Entropy,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )

    log.info(f"Quantized ONNX models saved to {output_path}")

    return output_path


def load_dataset_at(path: str):
    if path.startswith('/'):
        if (Path(path) / "metadata.jsonl").exists():
            return load_dataset("audiofolder", data_dir=path)["train"]
        return load_from_disk(path)["train"]
    else:
        return load_dataset(path)["train"]


class OnDemandSmartTurnDataset(Dataset):
    def __init__(self, hf_dataset, feature_extractor):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        audio_array = sample["audio"]["array"]

        audio_array = truncate_audio_to_last_n_seconds(audio_array, n_seconds=8)

        label = 1 if sample["endpoint_bool"] else 0

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
            max_length=8 * 16000,
            truncation=True,
            do_normalize=True,
        )

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "language": sample.get("language", "eng"),
            "dataset": sample.get("dataset", "unknown"),
            "midfiller": sample.get("midfiller", None),
            "endfiller": sample.get("endfiller", None),
        }


@dataclass
class SmartTurnDataCollator:
    def __call__(self, features: List[Dict[str, Union[torch.Tensor, str, None]]]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        batch = {
            "input_features": input_features,
            "labels": labels,
        }

        if "language" in features[0]:
            batch["language"] = [f["language"] for f in features]
        if "midfiller" in features[0]:
            batch["midfiller"] = [f["midfiller"] for f in features]
        if "endfiller" in features[0]:
            batch["endfiller"] = [f["endfiller"] for f in features]

        return batch


def prepare_datasets_ondemand(feature_extractor, config):
    log.info("Preparing datasets...")

    ensure_torchcodec_available()

    datasets_training = config["datasets_training"]
    datasets_test = config["datasets_test"]

    training_splits = []
    eval_splits = []
    test_splits = {}

    for dataset_path in datasets_training:
        dataset_name = dataset_path.split("/")[-1]
        log.info(f"Loading dataset '{dataset_name}'...")
        full_dataset = load_dataset_at(dataset_path)

        log.info("  |-> Splitting dataset into train/eval splits...")
        dataset_dict = full_dataset.train_test_split(test_size=0.1, seed=42)
        training_splits.append(dataset_dict["train"])
        eval_splits.append(dataset_dict["test"])

    log.info("Merging datasets...")

    merged_training_dataset = concatenate_datasets(training_splits).shuffle(seed=42)
    merged_eval_dataset = concatenate_datasets(eval_splits)

    log.info("Loading test datasets...")

    for dataset_path in datasets_test:
        dataset_name = dataset_path.split("/")[-1]
        test_dataset = load_dataset_at(dataset_path)
        test_splits[dataset_name] = test_dataset

    log.info("Wrapping datasets with OnDemandWhisperDataset...")
    wrapped_training = OnDemandSmartTurnDataset(merged_training_dataset, feature_extractor)
    wrapped_eval = OnDemandSmartTurnDataset(merged_eval_dataset, feature_extractor)
    wrapped_test_splits = {
        name: OnDemandSmartTurnDataset(dataset, feature_extractor)
        for name, dataset in test_splits.items()
    }
    wrapped_test_merged = (
        OnDemandSmartTurnDataset(
            concatenate_datasets(list(test_splits.values())).shuffle(seed=42),
            feature_extractor
        )
        if test_splits else None
    )

    return {
        "training": wrapped_training,
        "eval": wrapped_eval,
        "test": wrapped_test_splits,
        "test_merged": wrapped_test_merged,
        "raw_datasets": {
            "training": merged_training_dataset,
            "eval": merged_eval_dataset,
            "test": test_splits
        }
    }


def process_predictions(logits):
    """
    Converts raw logits into squeezed probability predictions and binary predictions.
    """
    if np.isnan(logits).any() or not np.isfinite(logits).all():
        raise ValueError("Non-finite or NaN values detected in logits during processing")

    probs = logits.squeeze()
    preds = (probs > 0.5).astype(int)

    return probs, preds


def get_predictions_and_labels(trainer, dataset, metric_key_prefix=None):
    """
    Returns tuple:
        - predictions: Raw prediction output from trainer
        - labels: Ground truth labels
        - probs: Squeezed probability predictions
        - preds: Binary predictions (probs > 0.5)
    """
    predictions = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)

    probs, preds = process_predictions(predictions.predictions)
    labels = predictions.label_ids

    return predictions, labels, probs, preds


class ExternalEvaluationCallback(TrainerCallback):

    def __init__(self, test_datasets, trainer):
        super().__init__()
        self.test_datasets = test_datasets
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        accuracies = {}
        language_metrics = {}
        midfiller_metrics = {}

        for dataset_name, dataset in self.test_datasets.items():
            predictions, labels, probs, preds = get_predictions_and_labels(
                self.trainer, dataset, f"exttest/{dataset_name}"
            )

            metrics = compute_metrics((probs, labels))

            external_metrics = {
                f"exttest/{dataset_name}_{k}": v
                for k, v in metrics.items()
            }

            external_metrics[f"exttest/{dataset_name}_prob_dist"] = wandb.Histogram(probs)

            external_metrics[f"train/global_step"] = state.global_step

            accuracies[dataset_name] = metrics["accuracy"]

            wandb.log(external_metrics)

            self._process_category_metrics(dataset, probs, labels, preds, language_metrics,
                                           column_name='language', default_value='unknown-error')
            self._process_category_metrics(dataset, probs, labels, preds, midfiller_metrics,
                                           column_name='midfiller', default_value='unknown')

        self._log_category_metrics(language_metrics, 'lang', state.global_step)
        self._log_category_metrics(midfiller_metrics, 'midfiller', state.global_step)

        if accuracies:
            lowest_accuracy = min(accuracies.values())
            lowest_accuracy_dataset = min(accuracies.keys(), key=lambda k: accuracies[k])

            accuracy_values = list(accuracies.values())
            mean_accuracy = sum(accuracy_values) / len(accuracy_values)

            wandb.log({
                "exttest/lowest_accuracy": lowest_accuracy,
                "exttest/lowest_accuracy_dataset": lowest_accuracy_dataset,
                "exttest/mean_accuracy": mean_accuracy,
                "exttest/accuracy_variance": np.var(accuracy_values),
                "train/global_step": state.global_step
            })

            log.info(f"Overall accuracy metrics:")
            log.info(f"  Lowest accuracy across all test datasets: {lowest_accuracy:.4f} ({lowest_accuracy_dataset})")
            log.info(f"  Mean accuracy: {mean_accuracy:.4f}")
            log.info(f"  Accuracy variance: {np.var(accuracy_values):.4f}")

    def _process_category_metrics(self, dataset, probs, labels, preds, category_metrics,
                                  column_name, default_value):
        if hasattr(dataset, 'dataset'):
            underlying_dataset = dataset.dataset
        else:
            underlying_dataset = dataset

        if hasattr(underlying_dataset, 'column_names') and column_name in underlying_dataset.column_names:
            categories = underlying_dataset[column_name]
        else:
            categories = [default_value] * len(dataset)

        for i, category in enumerate(categories):
            category_key = str(category).lower() if category is not None else default_value

            if category_key not in category_metrics:
                category_metrics[category_key] = {
                    'probs': [],
                    'labels': [],
                    'preds': []
                }

            category_metrics[category_key]['probs'].append(probs[i])
            category_metrics[category_key]['labels'].append(labels[i])
            category_metrics[category_key]['preds'].append(preds[i])

    def _log_category_metrics(self, category_metrics, metric_prefix, global_step):
        category_accuracies = {}

        for category, data in category_metrics.items():
            if len(data['labels']) == 0:
                continue

            cat_probs = np.array(data['probs'])
            cat_labels = np.array(data['labels'])

            metrics = compute_metrics((cat_probs, cat_labels))

            category_specific_metrics = {
                f"exttest/{metric_prefix}_{category}_{k}": v
                for k, v in metrics.items()
            }

            category_specific_metrics[f"exttest/{metric_prefix}_{category}_prob_dist"] = wandb.Histogram(cat_probs)
            category_specific_metrics[f"exttest/{metric_prefix}_{category}_sample_count"] = len(cat_labels)
            category_specific_metrics["train/global_step"] = global_step

            category_accuracies[category] = metrics["accuracy"]

            wandb.log(category_specific_metrics)

            log.info(f"{metric_prefix.capitalize()} {category} metrics: accuracy={metrics['accuracy']:.4f}, "
                     f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                     f"f1={metrics['f1']:.4f}, samples={len(cat_labels)}")

        if category_accuracies:
            min_accuracy = min(category_accuracies.values())
            max_accuracy = max(category_accuracies.values())
            mean_accuracy = sum(category_accuracies.values()) / len(category_accuracies)

            best_category = max(category_accuracies.keys(), key=lambda k: category_accuracies[k])
            worst_category = min(category_accuracies.keys(), key=lambda k: category_accuracies[k])

            summary_metrics = {
                f"exttest/{metric_prefix}_min_accuracy": min_accuracy,
                f"exttest/{metric_prefix}_max_accuracy": max_accuracy,
                f"exttest/{metric_prefix}_mean_accuracy": mean_accuracy,
                f"exttest/{metric_prefix}_accuracy_range": max_accuracy - min_accuracy,
                f"exttest/best_performing_{metric_prefix}": best_category,
                f"exttest/worst_performing_{metric_prefix}": worst_category,
                f"exttest/{metric_prefix}_categories_evaluated": len(category_accuracies),
                "train/global_step": global_step
            }

            if len(category_accuracies) > 1:
                summary_metrics[f"exttest/{metric_prefix}_accuracy_std"] = np.std(list(category_accuracies.values()))

            wandb.log(summary_metrics)

            category_type = metric_prefix.replace('_', ' ')
            log.info(f"{category_type.capitalize()} performance summary:")
            log.info(f"  Best performing {category_type}: {best_category} ({category_accuracies[best_category]:.4f})")
            log.info(
                f"  Worst performing {category_type}: {worst_category} ({category_accuracies[worst_category]:.4f})")
            log.info(f"  Mean accuracy across {category_type}s: {mean_accuracy:.4f}")
            log.info(f"  Accuracy range: {max_accuracy - min_accuracy:.4f}")

        if category_metrics:
            total_samples = sum(len(data['labels']) for data in category_metrics.values())
            distribution_metrics = {
                f"exttest/{metric_prefix}_{category}_percentage": (len(
                    category_metrics[category]['labels']) / total_samples) * 100
                for category in category_metrics.keys()
            }
            distribution_metrics["train/global_step"] = global_step
            wandb.log(distribution_metrics)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs, preds = process_predictions(logits)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division="warn"),
        "recall": recall_score(labels, preds, zero_division="warn"),
        "f1": f1_score(labels, preds, zero_division="warn"),
        "pred_positives": tp + fp,
        "pred_negatives": tn + fn,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def final_evaluate(trainer, dataset, split_name):
    log.info(f"Evaluating on {split_name} set...")
    metrics = trainer.evaluate(eval_dataset=dataset)

    predictions, labels, probs, preds = get_predictions_and_labels(trainer, dataset)

    wandb_metrics = {
        f"final/{split_name}_accuracy": metrics["eval_accuracy"],
        f"final/{split_name}_precision": metrics["eval_precision"],
        f"final/{split_name}_recall": metrics["eval_recall"],
        f"final/{split_name}_f1": metrics["eval_f1"],
    }

    wandb.log(wandb_metrics)

    return metrics, predictions


def do_training_run(run_name: str, output_dir: str = "./output"):
    log_dependencies()
    log_device_info()

    log.info(f"Starting training run: {run_name}")
    log.info(f"Output directory: {output_dir}")

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")

    wandb_run = wandb.init(
        project="speech-endpointing",
        name=run_name,
        config=CONFIG
    )

    wandb_run.define_metric(name="exttest/*", step_metric="train/global_step")

    model = SmartTurnV3Model.from_pretrained(CONFIG["base_model_name"], num_labels=1, ignore_mismatched_sizes=True)
    feature_extractor = WhisperFeatureExtractor(chunk_length=8) # 8 seconds

    freeze_encoder_layers = CONFIG.get("freeze_encoder_layers", 0)
    if freeze_encoder_layers > 0:
        for i, layer in enumerate(model.encoder.layers):
            if i < freeze_encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        log.info(f"Froze first {freeze_encoder_layers} encoder transformer layers")

    log_model_structure(model, CONFIG)

    datasets = prepare_datasets_ondemand(feature_extractor, CONFIG)

    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{run_name}",
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        num_train_epochs=CONFIG["num_epochs"],
        eval_strategy=IntervalStrategy.STEPS,
        gradient_accumulation_steps=1,
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        logging_steps=CONFIG["logging_steps"],
        load_best_model_at_end=CONFIG.get("load_best_model_at_end", False),
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type="cosine",
        report_to=["wandb"],
        dataloader_num_workers=6,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        tf32=False,
        disable_tqdm=True,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    log_dataset_statistics("training", datasets["training"])
    log_dataset_statistics("eval", datasets["eval"])

    for dataset_name, dataset in datasets["test"].items():
        log_dataset_statistics("test_" + dataset_name, dataset)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["training"],
        eval_dataset=datasets["eval"],
        compute_metrics=compute_metrics,
        data_collator=SmartTurnDataCollator(),
        callbacks=[
            ProgressLoggerCallback(log_interval=CONFIG["logging_steps"])
        ]
    )

    if datasets["test"]:
        trainer.add_callback(ExternalEvaluationCallback(
            test_datasets=datasets["test"],
            trainer=trainer
        ))

    log.info("Starting training...")
    trainer.train()

    final_save_path = f"{trainer.args.output_dir}/final_model"
    os.makedirs(final_save_path, exist_ok=True)

    feature_extractor.save_pretrained(final_save_path)
    trainer.save_model(final_save_path)

    export_path = os.path.join(final_save_path, "exports")
    os.makedirs(export_path, exist_ok=True)

    onnx_fp32_path = os.path.join(export_path, "model_fp32.onnx")

    trainer.model.eval().cpu()

    onnx_fp32_model_path = export_to_onnx_fp32(trainer.model, onnx_fp32_path, CONFIG)

    log.info(f"Training and export completed. Models saved to: {final_save_path}")

    wandb.finish()

    return onnx_fp32_model_path


def do_quantization_run(fp32_model_path: str):
    calibration_dataset_size = CONFIG["calibration_dataset_size"]

    log.info(f"Starting quantization run on {fp32_model_path} (calib dataset size {calibration_dataset_size})")

    feature_extractor = WhisperFeatureExtractor(chunk_length=8)  # 8 seconds

    datasets = prepare_datasets_ondemand(feature_extractor, CONFIG)

    parent_dir = os.path.dirname(fp32_model_path)

    quantized_onnx_path = quantize_onnx_model(
        onnx_fp32_path=fp32_model_path,
        training_dataset=datasets["training"],
        feature_extractor=feature_extractor,
        exports_path=parent_dir,
        calibration_dataset_size=calibration_dataset_size
    )

    return quantized_onnx_path


def do_benchmark_run(model_paths: List[str]):
    log.info(f"Benchmarking models: {model_paths}")

    feature_extractor = WhisperFeatureExtractor(chunk_length=8)  # 8 seconds

    dataset = prepare_datasets_ondemand(feature_extractor, CONFIG)["test_merged"]
    if dataset is None:
        raise ValueError("At least one test dataset is required for benchmarking")

    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace(".onnx", "")
        benchmark_path = os.path.join(os.path.dirname(model_path), "benchmarks")
        os.makedirs(benchmark_path, exist_ok=True)

        benchmark(
            onnx_path=model_path,
            run_description=model_name,
            dataset=dataset,
            limit=None,
            markdown_output=f"{benchmark_path}/{model_name}.md",
            batch_size=256
        )
