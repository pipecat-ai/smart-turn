import os
from datetime import datetime

import matplotlib.pyplot as plt
import modal
import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import Wav2Vec2Processor
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
from transformers.trainer_utils import IntervalStrategy
from transformers.training_args import TrainingArguments

from datasets import load_dataset, concatenate_datasets, load_from_disk
from logger import log, log_model_structure, log_dataset_statistics, log_dependencies, ProgressLoggerCallback
from model import Wav2Vec2ForEndpointing

# Define Modal stub and volume.
app = modal.App("endpointing-training")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

# Define Modal image with required dependencies.
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
    "wandb"
).add_local_python_source("logger").add_local_python_source("model")

# Hyperparameters and configuration
CONFIG = {
    "model_name": "facebook/wav2vec2-base-960h",

    # Three types of dataset are used during in this script: training, eval, and test.
    #
    # - The eval set is used to guide the training process, for example with early stopping, and selecting
    #   the best checkpoint.
    #
    # - The test set is kept completely separate from the training process, and is periodically used to
    #   evaluate the performance of the model.
    #
    # The datasets in `datasets_training` are split 80/10/10, and used for all three purposes.
    # The datasets in `datasets_test` are only used for testing, and are not split.
    #
    # All test datasets are stored and reported separately.
    "datasets_training": [
        "pipecat-ai/rime_2",
        "pipecat-ai/human_5_all",
        "pipecat-ai/human_convcollector_1",
        "pipecat-ai/orpheus_grammar_1",
        "pipecat-ai/orpheus_midfiller_1",
        "pipecat-ai/orpheus_endfiller_1",
        "pipecat-ai/chirp3_1",
    ],
    "datasets_test": [], # e.g. "/data/datasets/human_5_filler"

    # Training parameters
    "learning_rate": 5e-5,
    "num_epochs": 3,
    "train_batch_size": 30,
    "eval_batch_size": 64,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    # Evaluation parameters
    "eval_steps": 500,
    "save_steps": 500,
    "logging_steps": 100,
}


def load_dataset_at(path: str):
    # Ignore linter errors, this works fine
    if path.startswith('/'):
        return load_from_disk(path)["train"]
    else:
        return load_dataset(path)["train"]


def validate_audio_lengths(dataset, dataset_name):
    """Validate that all audio samples are between 0 and 16 seconds"""
    for i, sample in enumerate(dataset):
        audio_array = sample['audio']['array']

        duration = len(audio_array) / 16000

        if duration <= 0:
            raise ValueError(
                f"Fatal error: Audio sample {i} in dataset '{dataset_name}' has zero or negative length ({duration} seconds)")

        if duration > 16:
            raise ValueError(
                f"Fatal error: Audio sample {i} in dataset '{dataset_name}' exceeds 16 seconds limit ({duration} seconds)")

def prepare_datasets(preprocess_function, config):
    """
    Loads, splits, and organizes datasets based on config settings.

    Returns a dictionary with "training", "eval", and "test" entries.
    """
    datasets_training = config["datasets_training"]
    datasets_test = config["datasets_test"]

    overlap = set(datasets_training).intersection(set(datasets_test))
    if overlap:
        raise ValueError(f"Found overlapping datasets in training and test: {overlap}")

    training_splits = []
    eval_splits = []
    test_splits = {}

    for dataset_path in datasets_training:
        # Extract dataset name from path
        dataset_name = dataset_path.split("/")[-1]

        full_dataset = load_dataset_at(dataset_path)

        validate_audio_lengths(full_dataset, dataset_name)

        # Create train/eval/test split (80/10/10)
        dataset_dict = full_dataset.train_test_split(test_size=0.2, seed=42)
        training_splits.append(dataset_dict["train"])
        eval_test_dict = dataset_dict["test"].train_test_split(test_size=0.5, seed=42)

        eval_splits.append(eval_test_dict["train"])
        test_splits[dataset_name] = eval_test_dict["test"]

    # Merge training and eval splits
    merged_training_dataset = concatenate_datasets(training_splits).shuffle(seed=42)
    merged_eval_dataset = concatenate_datasets(eval_splits)

    # Load and add the full test datasets
    for dataset_path in datasets_test:
        dataset_name = dataset_path.split("/")[-1]
        test_dataset = load_dataset_at(dataset_path)

        validate_audio_lengths(test_dataset, dataset_name)

        test_splits[dataset_name] = test_dataset

    def apply_preprocessing(dataset):
        return dataset.map(
            preprocess_function,
            batched=True,
            batch_size=8,
            remove_columns=["audio", "endpoint_bool"],
            num_proc=16
        )

    merged_training_dataset = apply_preprocessing(merged_training_dataset)
    merged_eval_dataset = apply_preprocessing(merged_eval_dataset)

    for dataset_name, dataset in test_splits.items():
        test_splits[dataset_name] = apply_preprocessing(dataset)

    return {
        "training": merged_training_dataset,
        "eval": merged_eval_dataset,
        "test": test_splits
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

    def __init__(self, test_datasets, compute_metrics, trainer):
        super().__init__()
        self.test_datasets = test_datasets
        self.compute_metrics = compute_metrics
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):
        accuracies = {}  # Store accuracies for each dataset
        language_metrics = {}  # Store metrics aggregated by language
        midfiller_metrics = {}  # Store metrics aggregated by midfiller

        for dataset_name, dataset in self.test_datasets.items():
            predictions, labels, probs, preds = get_predictions_and_labels(
                self.trainer, dataset, f"exttest/{dataset_name}"
            )

            metrics = self.compute_metrics((probs, labels))

            external_metrics = {
                f"exttest/{dataset_name}_{k}": v
                for k, v in metrics.items()
            }

            # Create histogram for probability distribution
            external_metrics[f"exttest/{dataset_name}_prob_dist"] = wandb.Histogram(probs)

            external_metrics[f"train/global_step"] = state.global_step

            # Store accuracy for this dataset
            accuracies[dataset_name] = metrics["accuracy"]

            wandb.log(external_metrics)

            # Process category-based metrics
            self._process_category_metrics(dataset, probs, labels, preds, language_metrics,
                                           column_name='language', default_value='unknown-error')
            self._process_category_metrics(dataset, probs, labels, preds, midfiller_metrics,
                                           column_name='midfiller', default_value='unknown')

        # Log category-based metrics
        self._log_category_metrics(language_metrics, 'lang', state.global_step)
        self._log_category_metrics(midfiller_metrics, 'midfiller', state.global_step)

        if accuracies:
            # Log the lowest accuracy across all datasets
            lowest_accuracy = min(accuracies.values())
            lowest_accuracy_dataset = min(accuracies.keys(), key=lambda k: accuracies[k])

            # Calculate overall accuracy with penalty for poor performing datasets
            accuracy_values = list(accuracies.values())
            mean_accuracy = sum(accuracy_values) / len(accuracy_values)

            wandb.log({
                "exttest/lowest_accuracy": lowest_accuracy,
                "exttest/lowest_accuracy_dataset": lowest_accuracy_dataset,
                "exttest/mean_accuracy": mean_accuracy,
                "exttest/accuracy_variance": np.var(accuracy_values),
                "train/global_step": state.global_step
            })

            log.info(f"\nOverall accuracy metrics:")
            log.info(f"  Lowest accuracy across all test datasets: {lowest_accuracy:.4f} ({lowest_accuracy_dataset})")
            log.info(f"  Mean accuracy: {mean_accuracy:.4f}")
            log.info(f"  Accuracy variance: {np.var(accuracy_values):.4f}")

    def _log_language_metrics(self, language_metrics, global_step):
        """Compute and log metrics for each language."""
        language_accuracies = {}

        for lang, data in language_metrics.items():
            if len(data['labels']) == 0:
                continue

            # Convert to numpy arrays for metric computation
            lang_probs = np.array(data['probs'])
            lang_labels = np.array(data['labels'])
            lang_preds = np.array(data['preds'])

            # Compute metrics for this language
            metrics = self.compute_metrics((lang_probs, lang_labels))

            # Log language-specific metrics
            language_specific_metrics = {
                f"exttest/lang_{lang}_{k}": v
                for k, v in metrics.items()
            }

            # Add probability distribution histogram for this language
            language_specific_metrics[f"exttest/lang_{lang}_prob_dist"] = wandb.Histogram(lang_probs)
            language_specific_metrics[f"exttest/lang_{lang}_sample_count"] = len(lang_labels)
            language_specific_metrics["train/global_step"] = global_step

            # Store accuracy for cross-language analysis
            language_accuracies[lang] = metrics["accuracy"]

            wandb.log(language_specific_metrics)

            log.info(f"Language {lang} metrics: accuracy={metrics['accuracy']:.4f}, "
                     f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                     f"f1={metrics['f1']:.4f}, samples={len(lang_labels)}")

        # Log cross-language summary metrics
        if language_accuracies:
            min_lang_accuracy = min(language_accuracies.values())
            max_lang_accuracy = max(language_accuracies.values())
            mean_lang_accuracy = sum(language_accuracies.values()) / len(language_accuracies)

            # Find best and worst performing languages
            best_lang = max(language_accuracies.keys(), key=lambda k: language_accuracies[k])
            worst_lang = min(language_accuracies.keys(), key=lambda k: language_accuracies[k])

            wandb.log({
                "exttest/lang_min_accuracy": min_lang_accuracy,
                "exttest/lang_max_accuracy": max_lang_accuracy,
                "exttest/lang_mean_accuracy": mean_lang_accuracy,
                "exttest/lang_accuracy_range": max_lang_accuracy - min_lang_accuracy,
                "exttest/lang_accuracy_std": np.std(list(language_accuracies.values())),
                "exttest/best_performing_language": best_lang,
                "exttest/worst_performing_language": worst_lang,
                "exttest/languages_evaluated": len(language_accuracies),
                "train/global_step": global_step
            })

            log.info(f"\nLanguage performance summary:")
            log.info(f"  Best performing language: {best_lang} ({language_accuracies[best_lang]:.4f})")
            log.info(f"  Worst performing language: {worst_lang} ({language_accuracies[worst_lang]:.4f})")
            log.info(f"  Mean accuracy across languages: {mean_lang_accuracy:.4f}")
            log.info(f"  Accuracy range: {max_lang_accuracy - min_lang_accuracy:.4f}")

    def _process_category_metrics(self, dataset, probs, labels, preds, category_metrics,
                                  column_name, default_value):
        """Generic method to process and accumulate metrics by any categorical column."""
        # Check if the dataset has the specified column
        if column_name in dataset.column_names:
            categories = dataset[column_name]
        else:
            # Use default value if column doesn't exist
            categories = [default_value] * len(dataset)

        # Group by category
        for i, category in enumerate(categories):
            # Convert to string for consistency (handles booleans, etc.)
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
        """Generic method to compute and log metrics for any category grouping."""
        category_accuracies = {}

        for category, data in category_metrics.items():
            if len(data['labels']) == 0:
                continue

            # Convert to numpy arrays for metric computation
            cat_probs = np.array(data['probs'])
            cat_labels = np.array(data['labels'])

            # Compute metrics for this category
            metrics = self.compute_metrics((cat_probs, cat_labels))

            # Log category-specific metrics
            category_specific_metrics = {
                f"exttest/{metric_prefix}_{category}_{k}": v
                for k, v in metrics.items()
            }

            # Add probability distribution histogram for this category
            category_specific_metrics[f"exttest/{metric_prefix}_{category}_prob_dist"] = wandb.Histogram(cat_probs)
            category_specific_metrics[f"exttest/{metric_prefix}_{category}_sample_count"] = len(cat_labels)
            category_specific_metrics["train/global_step"] = global_step

            # Store accuracy for cross-category analysis
            category_accuracies[category] = metrics["accuracy"]

            wandb.log(category_specific_metrics)

            log.info(f"{metric_prefix.capitalize()} {category} metrics: accuracy={metrics['accuracy']:.4f}, "
                     f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
                     f"f1={metrics['f1']:.4f}, samples={len(cat_labels)}")

        # Log cross-category summary metrics
        if category_accuracies:
            min_accuracy = min(category_accuracies.values())
            max_accuracy = max(category_accuracies.values())
            mean_accuracy = sum(category_accuracies.values()) / len(category_accuracies)

            # Find best and worst performing categories
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

            # Add standard deviation for categories
            if len(category_accuracies) > 1:
                summary_metrics[f"exttest/{metric_prefix}_accuracy_std"] = np.std(list(category_accuracies.values()))

            wandb.log(summary_metrics)

            # Log summary information
            category_type = metric_prefix.replace('_', ' ')
            log.info(f"\n{category_type.capitalize()} performance summary:")
            log.info(f"  Best performing {category_type}: {best_category} ({category_accuracies[best_category]:.4f})")
            log.info(
                f"  Worst performing {category_type}: {worst_category} ({category_accuracies[worst_category]:.4f})")
            log.info(f"  Mean accuracy across {category_type}s: {mean_accuracy:.4f}")
            log.info(f"  Accuracy range: {max_accuracy - min_accuracy:.4f}")

        # Log category distribution percentages
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

    # Plot and save confusion matrix
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

    # Plot and save probability distribution
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

    # Log additional metrics to wandb
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
    gpu="L40S",
    memory=16384,
    cpu=16.0,
    volumes={"/data": volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def training_run(run_number):

    log_dependencies()

    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    CONFIG["run_name"] = f"v2-linearclassifier-{now}_run{run_number}"

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

    model = Wav2Vec2ForEndpointing.from_pretrained(CONFIG["model_name"], num_labels=1)
    processor = Wav2Vec2Processor.from_pretrained(CONFIG["model_name"])

    log_model_structure(model, CONFIG)

    def preprocess_function(batch):
        audio_arrays = [x["array"] for x in batch["audio"]]
        labels = [1 if lb else 0 for lb in batch["endpoint_bool"]]

        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length=16000 * 16,
            return_attention_mask=True,
            return_tensors="pt"
        )
        inputs["labels"] = labels

        inputs["language"] = batch["language"] if "language" in batch else (["eng"] * len(labels))
        if "midfiller" in batch:
            inputs["midfiller"] = batch["midfiller"]
        if "endfiller" in batch:
            inputs["endfiller"] = batch["endfiller"]
        
        return inputs

    datasets = prepare_datasets(preprocess_function, CONFIG)

    log_dataset_statistics("training", datasets["training"])
    log_dataset_statistics("eval", datasets["eval"])

    for dataset_name, dataset in datasets["test"].items():
        log_dataset_statistics("test_" + dataset_name, dataset)

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
        max_grad_norm=1.0,
        dataloader_num_workers=16,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        tf32=True,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["training"],
        eval_dataset=datasets["eval"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            ProgressLoggerCallback(log_interval=CONFIG["logging_steps"])
        ]
    )

    trainer.add_callback(ExternalEvaluationCallback(
        test_datasets=datasets["test"],
        compute_metrics=compute_metrics,
        trainer=trainer
    ))

    trainer.train()

    # Evaluate on validation set
    log.info(f"Final eval set evaluation:")
    evaluate_and_plot(trainer, datasets["eval"], "eval")

    # Evaluate on test set
    for dataset_name, dataset in datasets["test"].items():
        log.info(f"Test set evaluation ({dataset_name}):")
        evaluate_and_plot(trainer, dataset, dataset_name)

    # Save the final model and processor.
    final_save_path = f"{training_args.output_dir}/final_model"
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    log.info(f"Model saved to {final_save_path}")

    wandb.finish()

@app.local_entrypoint()
def main(run_number: str = "00"):
    training_run.remote(run_number)