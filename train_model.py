import os
import pandas as pd
import numpy as np
import torch
import librosa
from datasets import Dataset, load_metric
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)
from huggingface_hub import notebook_login, whoami


def preprocess_function(examples, feature_extractor, max_input_length):
    audio_arrays = [np.array(x["array"]).astype(np.float32) for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        return_attention_mask=True,
        padding="max_length",
        max_length=max_input_length
    )
    return inputs


def main():
    # Log in to Hugging Face
    notebook_login()

    # Load Dataset
    csv_path = "animals_sounds/animals_sounds.csv"
    df = pd.read_csv(csv_path)
    audio_folder = "animals_sounds"
    filepaths = df["filename"].apply(lambda x: os.path.join(audio_folder, x)).tolist()
    labels = df["category"].tolist()

    # Load and preprocess audio files
    def load_audio(path):
        array, _ = librosa.load(path, sr=16000)
        return {"array": array, "sampling_rate": 16000}

    dataset = Dataset.from_dict({
        "audio": [load_audio(path) for path in filepaths],
        "label": labels
    })

    # Split dataset into training and testing
    dataset = dataset.train_test_split(seed=42, test_size=0.1)
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    # Define Pretrained Model and Feature Extractor
    model_id = "facebook/hubert-base-ls960"
    username = whoami()["name"]
    model_name = f"{username}/wav2vec2-animal-sounds-finetuned"

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    max_input_length = 160000

    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, feature_extractor, max_input_length),
        remove_columns=["audio"],
        batched=True,
        batch_size=100
    )

    test_dataset = test_dataset.map(
        lambda x: preprocess_function(x, feature_extractor, max_input_length),
        remove_columns=["audio"],
        batched=True,
        batch_size=100
    )

    # Define Label Mappings
    categories = sorted(df["category"].unique().tolist())
    id2label = {i: label for i, label in enumerate(categories)}
    label2id = {label: i for i, label in id2label.items()}

    # Load and configure the model
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=len(categories),
        label2id=label2id,
        id2label=id2label
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=f"{model_name}-results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=True
    )

    # Metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics
    )

    # Train and Push Model to Hugging Face
    trainer.train()
    trainer.push_to_hub(
        finetuned_from=model_id,
        tasks="audio-classification",
        model_name="hubert-finetuned-animals"
    )


if __name__ == "__main__":
    main()
