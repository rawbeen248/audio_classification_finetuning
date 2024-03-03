import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer, TrainingArguments, Trainer
from huggingface_hub import notebook_login, whoami
from sklearn.metrics import accuracy_score

# Define the ESC50Dataset class
class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, base_path):
        self.base_path = base_path
        self.dataframe = pd.read_csv(csv_path)
        self.categories = sorted(self.dataframe['category'].unique().tolist())
        self.label2index = {label: index for index, label in enumerate(self.categories)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = os.path.join(self.base_path, row['filename'])
        audio, _ = librosa.load(audio_path, sr=16000)
        return audio, self.label2index[row['category']]

def main():
    # Step 1: Notebook Login
    notebook_login()

    # Step 2: Load Dataset
    audio_folder = 'animal_sounds'
    csv_file = 'animals_sounds.csv'
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['category'])
    train_dataset = ESC50Dataset(csv_path=train_df, base_path=audio_folder)
    val_dataset = ESC50Dataset(csv_path=val_df, base_path=audio_folder)

    # Step 3: Prepare the Model
    username = whoami()["name"]
    model_name = f"{username}/wav2vec2-animal-sounds-finetuned"
    model_id = "facebook/wav2vec2-base-960h"
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id, num_labels=len(train_dataset.categories))
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_id)

    # Step 4: Define Training Arguments and Metrics
    training_args = TrainingArguments(
        output_dir=f"{model_name}-results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=10,
        push_to_hub=True,
        push_to_hub_model_id=model_name
    )

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

    # Step 5: Train and Push to Hub
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    kwargs = {
        "finetuned_from": model_id,
        "tasks": "audio-classification",
        "dataset": "ESC-50 Animals Subset"
    }

    trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    main()
