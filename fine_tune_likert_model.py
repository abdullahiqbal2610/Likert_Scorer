"""
Fine-tune DistilBERT for Likert Scale Classification (1-5)
FIXED VERSION with better base model and training setup
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os

# Configuration - FIXED
MODEL_NAME = "distilbert-base-uncased"  # Changed from SST-2 model
OUTPUT_DIR = "./likert_model_finetuned"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 10  # Increased epochs for small dataset
LEARNING_RATE = 3e-5  # Slightly higher learning rate

class LikertDataset(Dataset):
    """Custom dataset for Likert scale classification"""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label - 1, dtype=torch.long)  # Convert 1-5 to 0-4
        }

def compute_metrics(pred):
    """Compute accuracy and F1 score"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    return {
        'accuracy': acc,
        'f1': f1
    }

def load_and_prepare_data(csv_path):
    """Load dataset and split into train/val/test"""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Total samples: {len(df)}")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

    # Check if distribution is balanced
    label_counts = df['label'].value_counts()
    print(f"\nSamples per class: min={label_counts.min()}, max={label_counts.max()}")

    # Split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df

def main():
    """Main training pipeline"""

    print("="*60)
    print("LIKERT SCALE MODEL FINE-TUNING - FIXED VERSION")
    print("="*60)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    train_df, val_df, test_df = load_and_prepare_data('likert_training_data_3000.csv')

    # Load tokenizer from base model
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

    # Create datasets
    print("Creating datasets...")
    train_dataset = LikertDataset(
        train_df['text'].values,
        train_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )

    val_dataset = LikertDataset(
        val_df['text'].values,
        val_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )

    test_dataset = LikertDataset(
        test_df['text'].values,
        test_df['label'].values,
        tokenizer,
        MAX_LENGTH
    )

    # Load model with proper config
    print(f"\nLoading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,  # 5 classes: 1, 2, 3, 4, 5
    )

    # Training arguments - IMPROVED
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,  # Reduced for small dataset
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        learning_rate=LEARNING_RATE,
        save_total_limit=3,
        seed=42,
        # Add gradient accumulation for small batch size
        gradient_accumulation_steps=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    trainer.train()

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET...")
    print("="*60)
    test_results = trainer.predict(test_dataset)

    # Get predictions
    predictions = test_results.predictions.argmax(-1) + 1  # Convert back to 1-5
    true_labels = test_df['label'].values

    # Print results
    print("\n=== TEST SET RESULTS ===")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"F1 Score (weighted): {f1_score(true_labels, predictions, average='weighted'):.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(
        true_labels,
        predictions,
        target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']
    ))

    # Confusion matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(true_labels, predictions)
    print("Rows=True, Cols=Predicted")
    print(cm)

    # Save model
    print(f"\n=== SAVING MODEL ===")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to: {OUTPUT_DIR}")

    # Save some example predictions
    print("\n=== EXAMPLE PREDICTIONS (Showing Errors) ===")
    errors = 0
    for i in range(len(test_df)):
        text = test_df.iloc[i]['text']
        true = true_labels[i]
        pred = predictions[i]

        if true != pred:
            errors += 1
            print(f"✗ Text: {text[:70]}...")
            print(f"   True: {true}, Predicted: {pred}\n")
            if errors >= 10:
                break

    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel path: {OUTPUT_DIR}")
    print(f"Test Accuracy: {accuracy_score(true_labels, predictions):.2%}")

if __name__ == "__main__":
    main()