import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import modal
import numpy as np
import onnx
import onnxruntime as ort
import seaborn as sns
import torch
import wandb
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, quant_pre_process, \
    QuantFormat, CalibrationMethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.ao.quantization import disable_fake_quant, disable_observer, FakeQuantize, FusedMovingAvgObsFakeQuantize
from torch.nn.functional import softmax
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor, WhisperPreTrainedModel, WhisperConfig
# noinspection PyProtectedMember
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from datasets import load_dataset, concatenate_datasets, load_from_disk
from logger import log, log_model_structure, log_dataset_statistics, log_dependencies, ProgressLoggerCallback

app = modal.App("endpointing-training")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers[torch]==4.48.2",
    "datasets==3.2.0",
    "scikit-learn==1.6.1",
    "seaborn",
    "matplotlib",
    "numpy",
    "librosa",
    "soundfile",
    "wandb",
    "torchaudio",
    "onnx",
    "onnxruntime",
).add_local_python_source("logger")

CONFIG = {
    "model_name": "openai/whisper-tiny",

    "datasets_training": ["/data/datasets/smart-turn-data-v3-train"],
    "datasets_test": ["/data/datasets/smart-turn-data-v3-test"],

    "learning_rate": 5e-5,
    "num_epochs": 4, # overfitting starts to occur beyond 4 epochs
    "train_batch_size": 384,
    "eval_batch_size": 128,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,

    "qat_start_epoch": 1,
    "quantization_backend": "fbgemm",

    "onnx_opset_version": 20,
    "calibration_dataset_size": 256,
}

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

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

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
    """Calibration dataset for ONNX quantization with stratified sampling"""

    def __init__(self, dataset, feature_extractor, max_samples=500):
        self.feature_extractor = feature_extractor

        if hasattr(dataset, 'dataset'):
            # Ensure balanced sampling of endpoint classes
            underlying = dataset.dataset
            positive_indices = [i for i, sample in enumerate(underlying) if sample["endpoint_bool"]]
            negative_indices = [i for i, sample in enumerate(underlying) if not sample["endpoint_bool"]]

            # Sample half from each class
            import random
            random.seed(42)
            pos_sample_size = min(max_samples // 2, len(positive_indices))
            neg_sample_size = min(max_samples - pos_sample_size, len(negative_indices))

            selected_indices = (random.sample(positive_indices, pos_sample_size) +
                                random.sample(negative_indices, neg_sample_size))
            self.indices = selected_indices[:max_samples]
            self.dataset = underlying

            log.info(
                f"Calibration dataset: {pos_sample_size} positive + {neg_sample_size} negative = {len(self.indices)} total samples")
        else:
            raise ValueError("Invalid dataset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]

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


class QuantizationAwareTrainer(Trainer):
    """Custom trainer with quantization aware training support"""

    def __init__(self, *args, qat_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_config = qat_config
        self.qat_enabled = False

    def _prepare_model_for_qat(self):
        if not self.qat_enabled:
            torch.backends.quantized.engine = self.qat_config["quantization_backend"]

            qconfig = torch.quantization.get_default_qat_qconfig(self.qat_config["quantization_backend"])

            self.model.qconfig = qconfig

            embedding_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    log.info(f"Excluding embedding layer from quantization: {name}")
                    module.qconfig = None
                    embedding_count += 1

            layernorm_count = 0
            for name, module in self.model.named_modules():
                if isinstance(module,
                              (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    log.info(f"Excluding normalization layer from quantization: {name}")
                    module.qconfig = None
                    layernorm_count += 1

            torch.quantization.prepare_qat(self.model, inplace=True)

            self.qat_enabled = True

            log.info("Model prepared for quantization aware training")
            log.info(
                f"Excluded {embedding_count} embedding layers and {layernorm_count} normalization layers from quantization")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to handle QAT preparation"""
        if not self.qat_enabled and self.state.epoch >= self.qat_config["qat_start_epoch"]:
            self._prepare_model_for_qat()

        return super().training_step(model, inputs)


def export_to_onnx_fp32(model, example_input, output_path, config):
    """Export QAT model to ONNX FP32 format"""
    try:
        log.info("Exporting QAT model to ONNX FP32...")

        class ONNXExportWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, input_features):
                out = self.inner(input_features)
                return out["logits"] if isinstance(out, dict) else out

        export_model = ONNXExportWrapper(model).eval().cpu()

        disable_fake_quant(export_model)
        disable_observer(export_model)

        export_model = copy.deepcopy(export_model)

        def _strip_qat(m: torch.nn.Module):
            for name, child in list(m.named_children()):
                # recurse first
                _strip_qat(child)
                # replace fake-quant observers with identity
                if isinstance(child, (FakeQuantize, FusedMovingAvgObsFakeQuantize)):
                    setattr(m, name, torch.nn.Identity())
                # convert any QAT module that implements to_float()
                elif hasattr(child, "to_float") and callable(getattr(child, "to_float")):
                    try:
                        setattr(m, name, child.to_float())
                    except Exception:
                        pass

        _strip_qat(export_model)

        torch.onnx.export(
            export_model,
            example_input,
            output_path,
            export_params=True,
            opset_version=config["onnx_opset_version"],
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['logits'],
            dynamic_axes={'input_features': {0: 'batch_size'}, 'logits': {0: 'batch_size'}},
            verbose=False
        )

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        log.info(f"FP32 ONNX model saved to {output_path}")
        return output_path

    except Exception as e:
        log.error(f"Failed to export to ONNX: {e}")
        return None


def quantize_onnx_model(onnx_fp32_path, calibration_dataset, output_path):
    """Quantize ONNX model using static quantization"""
    try:
        log.info("Quantizing ONNX model to INT8...")

        pre_path = output_path.replace(".onnx", "_pre.onnx")
        quant_pre_process(
            onnx_fp32_path,
            pre_path,
            skip_optimization=False,  # let it fold/clean
            disable_shape_inference=False
        )
        # Calibrate + quantize
        quantize_static(
            model_input=pre_path,
            model_output=output_path,
            calibration_data_reader=ONNXCalibrationDataReader(calibration_dataset),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv", "MatMul", "Gemm"]
        )

        log.info(f"Quantized ONNX model saved to {output_path}")

        # Verify the quantized model
        quantized_model = onnx.load(output_path)
        onnx.checker.check_model(quantized_model)

        return output_path

    except Exception as e:
        log.error(f"Failed to quantize ONNX model: {e}")
        return None


def load_dataset_at(path: str):
    if path.startswith('/'):
        return load_from_disk(path)["train"]
    else:
        return load_dataset(path)["train"]


def truncate_audio_to_last_n_seconds(audio_array, n_seconds=8, sample_rate=16000):
    max_samples = n_seconds * sample_rate
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array


class OnDemandWhisperDataset(Dataset):
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
            "midfiller": sample.get("midfiller", None),
            "endfiller": sample.get("endfiller", None),
        }


@dataclass
class WhisperDataCollator:
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
    wrapped_training = OnDemandWhisperDataset(merged_training_dataset, feature_extractor)
    wrapped_eval = OnDemandWhisperDataset(merged_eval_dataset, feature_extractor)
    wrapped_test_splits = {
        name: OnDemandWhisperDataset(dataset, feature_extractor)
        for name, dataset in test_splits.items()
    }

    return {
        "training": wrapped_training,
        "eval": wrapped_eval,
        "test": wrapped_test_splits,
        "raw_datasets": {
            "training": merged_training_dataset,
            "eval": merged_eval_dataset,
            "test": test_splits
        }
    }


def save_dataset_ids(datasets, output_dir):
    ids_dict = {}

    if 'id' in datasets["raw_datasets"]["training"].column_names:
        train_ids = [id for id in datasets["raw_datasets"]["training"]["id"] if id is not None]
        ids_dict["train"] = sorted(train_ids)

    if 'id' in datasets["raw_datasets"]["eval"].column_names:
        eval_ids = [id for id in datasets["raw_datasets"]["eval"]["id"] if id is not None]
        ids_dict["eval"] = sorted(eval_ids)

    ids_dict["test"] = {}
    for name, dataset in datasets["raw_datasets"]["test"].items():
        if 'id' in dataset.column_names:
            test_ids = [id for id in dataset["id"] if id is not None]
            ids_dict["test"][name] = sorted(test_ids)

    ids_path = os.path.join(output_dir, "dataset_ids.json")
    with open(ids_path, 'w') as f:
        json.dump(ids_dict, f, indent=2)

    log.info(f"Saved dataset IDs to {ids_path}")
    return ids_path


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


def evaluate_and_plot(trainer, dataset, split_name):
    log.info(f"Evaluating on {split_name} set...")
    metrics = trainer.evaluate(eval_dataset=dataset)

    predictions, labels, probs, preds = get_predictions_and_labels(trainer, dataset)

    output_dir = os.path.join(trainer.args.output_dir, "evaluation_plots")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    try:
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Incomplete', 'Complete'],
                    yticklabels=['Incomplete', 'Complete'])
        plt.title(f'Confusion Matrix - {split_name.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{split_name}.png')
        plt.savefig(confusion_matrix_path)
        plt.close()
        log.info(f"Saved confusion matrix to {confusion_matrix_path}")
    except Exception as e:
        log.error(f"Could not create confusion matrix for {split_name}: {e}")
        confusion_matrix_path = None

    plt.figure(figsize=(10, 6))
    try:
        plt.hist(probs, bins=50, alpha=0.5, label='Probability of Complete')
        plt.title(f'Distribution of Completion Probabilities - {split_name.capitalize()} Set')
        plt.xlabel('Probability of Complete')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        prob_dist_path = os.path.join(output_dir, f'probability_distribution_{split_name}.png')
        plt.savefig(prob_dist_path)
        plt.close()
        log.info(f"Saved probability distribution to {prob_dist_path}")
    except Exception as e:
        log.error(f"Could not create probability distribution for {split_name}: {e}")
        prob_dist_path = None

    wandb_metrics = {
        f"final/{split_name}_accuracy": metrics["eval_accuracy"],
        f"final/{split_name}_precision": metrics["eval_precision"],
        f"final/{split_name}_recall": metrics["eval_recall"],
        f"final/{split_name}_f1": metrics["eval_f1"],
    }

    if confusion_matrix_path:
        wandb_metrics[f"final/confusion_matrix_{split_name}"] = wandb.Image(confusion_matrix_path)
    if prob_dist_path:
        wandb_metrics[f"final/probability_distribution_{split_name}"] = wandb.Image(prob_dist_path)

    wandb.log(wandb_metrics)

    return metrics, predictions


@app.function(
    image=image,
    gpu="L4",
    memory=8192,
    cpu=6.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def training_run(run_number):
    log_dependencies()

    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    CONFIG["run_name"] = f"v3-{now}_run{run_number}"

    log.info(f"Starting training run: {CONFIG['run_name']}")

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")

    wandb_run = wandb.init(
        project="speech-endpointing",
        name=CONFIG["run_name"],
        config=CONFIG
    )

    wandb_run.define_metric(name="exttest/*", step_metric="train/global_step")

    model = SmartTurnV3Model.from_pretrained(CONFIG["model_name"], num_labels=1, ignore_mismatched_sizes=True)
    feature_extractor = WhisperFeatureExtractor(chunk_length=8) # 8 seconds

    log_model_structure(model, CONFIG)

    datasets = prepare_datasets_ondemand(feature_extractor, CONFIG)

    training_args = TrainingArguments(
        output_dir=f"/data/output/{CONFIG['run_name']}",
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        num_train_epochs=CONFIG["num_epochs"],
        eval_strategy=IntervalStrategy.STEPS,
        gradient_accumulation_steps=1,
        eval_steps=CONFIG["eval_steps"],
        save_steps=CONFIG["save_steps"],
        logging_steps=CONFIG["logging_steps"],
        load_best_model_at_end=True,
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
        tf32=True,
        disable_tqdm=True,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    save_dataset_ids(datasets, training_args.output_dir)

    log_dataset_statistics("training", datasets["training"])
    log_dataset_statistics("eval", datasets["eval"])

    for dataset_name, dataset in datasets["test"].items():
        log_dataset_statistics("test_" + dataset_name, dataset)

    # Use custom quantization-aware trainer
    trainer = QuantizationAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["training"],
        eval_dataset=datasets["eval"],
        compute_metrics=compute_metrics,
        data_collator=WhisperDataCollator(),
        qat_config=CONFIG,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            ProgressLoggerCallback(log_interval=CONFIG["logging_steps"])
        ]
    )

    trainer.add_callback(ExternalEvaluationCallback(
        test_datasets=datasets["test"],
        trainer=trainer
    ))

    # Train the model
    log.info("Starting training...")
    trainer.train()

    final_save_path = f"{trainer.args.output_dir}/final_model"
    os.makedirs(final_save_path, exist_ok=True)

    feature_extractor.save_pretrained(final_save_path)
    trainer.save_model(final_save_path)

    export_path = os.path.join(final_save_path, "exports")
    os.makedirs(export_path, exist_ok=True)

    example_input = torch.randn(1, 80, 800)

    onnx_fp32_path = os.path.join(export_path, "model_fp32.onnx")
    onnx_int8_path = os.path.join(export_path, "model_int8.onnx")

    # Export the QAT model directly (before torch.quantization.convert)
    # This preserves the fake quantization nodes that ONNX can handle
    trainer.model.eval().cpu()

    onnx_fp32_model_path = export_to_onnx_fp32(trainer.model, example_input, onnx_fp32_path, CONFIG)

    if onnx_fp32_model_path is not None:
        log.info("ONNX FP32 export successful")
        wandb.log({"export/onnx_fp32_success": True})

        calibration_dataset = CalibrationDataset(
            datasets["training"],
            feature_extractor,
            max_samples=CONFIG["calibration_dataset_size"],
        )

        # Quantize ONNX model to INT8
        quantized_onnx_path = quantize_onnx_model(onnx_fp32_model_path, calibration_dataset, onnx_int8_path)

        if quantized_onnx_path is not None:
            log.info("ONNX INT8 quantization successful")
            wandb.log({"export/onnx_int8_success": True})

            # Test the quantized ONNX model
            try:
                session = ort.InferenceSession(quantized_onnx_path)
                test_input = calibration_dataset[0].reshape(1, 80, 800).astype(np.float32)
                outputs = session.run(None, {"input_features": test_input})
                log.info(f"ONNX model test successful. Output shape: {outputs[0].shape}")
                wandb.log({"export/onnx_test_success": True})
            except Exception as e:
                log.error(f"ONNX model test failed: {e}")
                wandb.log({"export/onnx_test_success": False})
        else:
            log.error("ONNX INT8 quantization failed")
            wandb.log({"export/onnx_int8_success": False})
    else:
        log.error("ONNX FP32 export failed")
        wandb.log({"export/onnx_fp32_success": False})

    # Now convert the model to fully quantized format for PyTorch inference
    trainer.model.eval().cpu()
    quantized_model = torch.quantization.convert(copy.deepcopy(trainer.model), inplace=False)

    # Save quantized PyTorch model
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'model_config': CONFIG,
    }, os.path.join(final_save_path, "quantized_model.pth"))

    log.info(f"Training and export completed. Models saved to: {final_save_path}")

    wandb.finish()


@app.local_entrypoint()
def main(run_number: str = "00"):
    training_run.remote(run_number)
