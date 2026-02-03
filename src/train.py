import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from transformers import (
    DistilBertTokenizerFast,                # tokenizer                
    DistilBertForSequenceClassification,    # model
    Trainer,                                # handles training loop
    TrainingArguments                       # configures training      
)

# Configuration
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/cleaned_data.csv"
MAX_LENGTH = 128 # tweets truncated/padded to 128 tokens (our tweets are short so it works)
BATCH_SIZE = 16 
EPOCHS = 3 # number of passes through the training data
SEED = 42 

# Load cleaned data
df = pd.read_csv(DATA_PATH)

texts = df["text"].tolist()
labels = df["target"].tolist()


# Train / Validation split (test split is created but NOT used here)
# 70% will go to training, 30% to a temporary bucket that will be split equally into validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    texts,
    labels,
    test_size=0.3,
    random_state=SEED,
    stratify=labels # keep same proportion in each split
)

# Further splits the 30% into 15% validation and 15% test
X_val, _, y_val, _ = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_temp
)

# Tokenization

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

train_encodings = tokenize(X_train)
val_encodings = tokenize(X_val)


# Dataset class
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, y_train)
val_dataset = TweetDataset(val_encodings, y_val)


# Model
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


# Metric (F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds)}


# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",                
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=SEED,
    logging_steps=100,
    save_total_limit=3
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# Training
trainer.train()


# Save model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

print("Training complete. Model saved to ./final_model")
