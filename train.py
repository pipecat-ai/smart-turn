import os
from datetime import datetime

import matplotlib.pyplot as plt
import modal
import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoFeatureExtractor,
    Wav2Vec2BertForSequenceClassification,
    EarlyStoppingCallback
)

from datasets import load_dataset, concatenate_datasets, load_from_disk

# Define Modal stub and volume.
app = modal.App("endpointing-training")
volume = modal.Volume.from_name("endpointing", create_if_missing=False)

# Define Modal image with required dependencies.
image = modal.Image.debian_slim().apt_install("ffmpeg").pip_install(
    "torch",
    "transformers[torch]",
    "datasets",
    "scikit-learn",
    "seaborn",
    "matplotlib",
    "numpy",
    "librosa==0.9.2",
    "soundfile",
    "wandb"
)

# Hyperparameters and configuration
CONFIG = {
    "run_name": "model-v1",
    "model_name": "facebook/w2v-bert-2.0",

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
    "datasets_training": ["pipecat-ai/rime_2", "pipecat-ai/human_5_all"],
    "datasets_test": [], # e.g. "/data/datasets/human_5_filler"

    # Training parameters
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "train_batch_size": 12,
    "eval_batch_size": 32,
    "warmup_ratio": 0.2,
    "weight_decay": 0.05,
    "gradient_accumulation_steps": 1,

    # Evaluation parameters
    "eval_steps": 50,
    "save_steps": 50,
    "logging_steps": 5,

    # Model architecture parameters
    "num_frozen_layers": 20
}

def load_dataset_at(path: str):
    if path.startswith('/'):
        return load_from_disk(path)["train"]
    else:
        return load_dataset(path)["train"]

def prepare_datasets(preprocess_function):
    """
    Loads, splits, and organizes datasets based on CONFIG settings.

    Returns a dictionary with "training", "eval", and "test" entries.
    """
    datasets_training = CONFIG["datasets_training"]
    datasets_test = CONFIG["datasets_test"]

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
        test_splits[dataset_name] = load_dataset_at(dataset_path)

    # Apply preprocessing function

    merged_training_dataset = merged_training_dataset.map(
        preprocess_function,
        batched=False,
        remove_columns=merged_training_dataset.column_names
    )

    merged_eval_dataset = merged_eval_dataset.map(
        preprocess_function,
        batched=False,
        remove_columns=merged_eval_dataset.column_names
    )

    for dataset_name, dataset in test_splits.items():
        test_splits[dataset_name] = dataset.map(
            preprocess_function,
            batched=False,
            remove_columns=dataset.column_names
        )

    return {
        "training": merged_training_dataset,
        "eval": merged_eval_dataset,
        "test": test_splits
    }


class ExternalEvaluationCallback(TrainerCallback):

    def __init__(self, test_datasets, compute_metrics, trainer):
        super().__init__()
        self.test_datasets = test_datasets
        self.compute_metrics = compute_metrics
        self.trainer = trainer

    def on_evaluate(self, args, state, control, **kwargs):

        for dataset_name, dataset in self.test_datasets.items():

            predictions = self.trainer.predict(dataset, metric_key_prefix=f"test_{dataset_name}")
            probs = predictions.predictions
            labels = predictions.label_ids

            metrics = self.compute_metrics((probs, labels))

            external_metrics = {
                f"test_{dataset_name}_{k}": v
                for k, v in metrics.items()
            }

            external_metrics[f"test_{dataset_name}_prob_dist"] = wandb.Histogram(probs.squeeze())

            wandb.log(external_metrics, step=state.global_step)

            print(f"\nExternal Evaluation Metrics ({dataset_name}):")
            for k, v in external_metrics.items():
                if isinstance(v, (float, int)):
                    print(f"{k}: {v:.4f}")

def log_dataset_statistics(split_name, dataset):
    """Log detailed statistics about each dataset split."""

    print(f"\n-- Dataset statistics: {split_name} --")

    # Basic statistics
    total_samples = len(dataset)
    if "labels" in dataset.features:
        labels = dataset["labels"]
        positive_samples = sum(1 for label in labels if label == 1)
        negative_samples = total_samples - positive_samples
        positive_ratio = positive_samples / total_samples * 100

        print(f"  Total samples: {total_samples:,}")
        print(f"  Positive samples (Complete): {positive_samples:,} ({positive_ratio:.2f}%)")
        print(f"  Negative samples (Incomplete): {negative_samples:,} ({100 - positive_ratio:.2f}%)")

        # Audio length statistics if available
        if "audio" in dataset.features:
            audio_lengths = [len(x["array"]) / 16000 for x in dataset["audio"]]  # Convert to seconds
            avg_length = sum(audio_lengths) / len(audio_lengths)
            min_length = min(audio_lengths)
            max_length = max(audio_lengths)

            print(f"  Audio statistics (in seconds):")
            print(f"    Average length: {avg_length:.2f}")
            print(f"    Min length: {min_length:.2f}")
            print(f"    Max length: {max_length:.2f}")
    else:
        print(f"  (no labels!)")
        print(f"  Total samples: {total_samples:,}")

@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    timeout=20000,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})
    ]
)
def training_run():
    # Initialize Weights & Biases.
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")
    wandb.init(project="speech-endpointing", name=CONFIG["run_name"], config=CONFIG)

    # Initialize model and processor using the W2v-BERT 2.0 checkpoint.
    model = Wav2Vec2BertForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=2)
    processor = AutoFeatureExtractor.from_pretrained(CONFIG["model_name"])

    # Freeze lower layers of the encoder in the wav2vec2_bert backbone.
    encoder_layers = model.wav2vec2_bert.encoder.layers

    for layer_idx in range(CONFIG["num_frozen_layers"]):
        for param in encoder_layers[layer_idx].parameters():
            param.requires_grad = False

    # Define a preprocessing function that processes one example at a time.
    def preprocess_function(example):
        # Extract the audio array.
        audio_array = example["audio"]["array"]
        label = 1 if example["endpoint_bool"] else 0

        # print("Audio array shape:", audio_array.shape)

        inputs = processor(
            audio_array,
            sampling_rate=16000,
            padding="max_length",
            truncation=True,
            max_length= 800, # raw sample length * sample rate / downsample factor
            return_attention_mask=True,
            return_tensors="pt"
        )

        # Remove extra batch dimension (if necessary)
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)

        # Add the label.
        inputs["labels"] = label
        return inputs

    datasets = prepare_datasets(preprocess_function)

    log_dataset_statistics("training", datasets["training"])
    log_dataset_statistics("eval", datasets["eval"])

    for dataset_name, dataset in datasets["test"].items():
        log_dataset_statistics(dataset_name, dataset)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0)
        }

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        metrics.update({
            "pred_positives": tp + fp,
            "pred_negatives": tn + fn,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        })

        return metrics

    def evaluate_and_plot(trainer, dataset, split_name):
        print(f"\nEvaluating on {split_name} set...")
        metrics = trainer.evaluate(eval_dataset=dataset)

        predictions = trainer.predict(dataset)
        logits = predictions.predictions # shape: (num_samples, num_classes)
        preds = np.argmax(logits, axis=1) # shape: (num_samples,)
        labels = predictions.label_ids

        output_dir = os.path.join(trainer.args.output_dir, "evaluation_plots")
        os.makedirs(output_dir, exist_ok=True)

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(labels, preds), annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Incomplete', 'Complete'],
                    yticklabels=['Incomplete', 'Complete'])
        plt.title(f'Confusion Matrix - {split_name.capitalize()} Set')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{split_name}.png')
        plt.savefig(confusion_matrix_path)
        wandb.log({f"confusion_matrix_{split_name}": wandb.Image(confusion_matrix_path)})
        plt.close()
        print(f"Saved confusion matrix to {confusion_matrix_path}")

        # Plot and save probability distribution
        plt.figure(figsize=(10, 6))
        # plt.hist(probs.squeeze(), bins=50, alpha=0.5, label='All Samples')
        # plt.hist(probs.squeeze()[labels == 1], bins=50, alpha=0.5, label='True Complete')
        plt.hist(logits[:,1], bins=50, alpha=0.5, label='Probability of Class 1')
        plt.title(f'Distribution of Completion Probabilities - {split_name.capitalize()} Set')
        plt.xlabel('Probability of Complete')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        prob_dist_path = os.path.join(output_dir, f'probability_distribution_{split_name}.png')
        plt.savefig(prob_dist_path)
        wandb.log({f"probability_distribution_{split_name}": wandb.Image(prob_dist_path)})
        plt.close()
        print(f"Saved probability distribution to {prob_dist_path}")

        # Log additional metrics to wandb
        wandb.log({
            f"{split_name}_accuracy": metrics["eval_accuracy"],
            f"{split_name}_precision": metrics["eval_precision"],
            f"{split_name}_recall": metrics["eval_recall"],
            f"{split_name}_f1": metrics["eval_f1"]
        })

        return metrics, predictions

    # Set training arguments.
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    training_args = TrainingArguments(
        output_dir=f"/data/output/{CONFIG['run_name']}-{current_time}",
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        num_train_epochs=CONFIG["num_epochs"],
        evaluation_strategy="steps",
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
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
        dataloader_num_workers=5,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        tf32=True,
        fp16=True,
    )

    # Instantiate the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["training"],
        eval_dataset=datasets["eval"],
        tokenizer=processor,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
        ]
    )

    trainer.add_callback(ExternalEvaluationCallback(
        test_datasets=datasets["test"],
        compute_metrics=compute_metrics,
        trainer=trainer
    ))

    def log_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Train the model
    print(f"\n[{log_timestamp()}] Starting training...")
    trainer.train()

    # Evaluate on validation set
    print(f"\n[{log_timestamp()}] Final eval set evaluation:")
    evaluate_and_plot(trainer, datasets["eval"], "eval")

    # Evaluate on test set
    for dataset_name, dataset in datasets["test"].items():
        print(f"\n[{log_timestamp()}] Test set evaluation ({dataset_name}):")
        evaluate_and_plot(trainer, dataset, dataset_name)

    # Save the final model and processor.
    final_save_path = f"{training_args.output_dir}/final_model"
    trainer.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"\nModel saved to {final_save_path}")

    wandb.finish()

@app.local_entrypoint()
def main():
    training_run.remote()
