import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, classification_report

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer

# ---------------- CONFIG ----------------
MODEL_DIR = "./final_model"
DATA_PATH = "BSDSA_cleaned_data.csv"
MAX_LENGTH = 128
SEED = 42
# ----------------------------------------

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load data
df = pd.read_csv(DATA_PATH)
texts = df["text"].tolist()
labels = df["target"].tolist()

# Recreate EXACT splits
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.3, random_state=SEED, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
)

# Load tokenizer and model (NO TRAINING)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

# Tokenize test set
test_encodings = tokenizer(
    X_test,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH
)

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

test_dataset = TweetDataset(test_encodings, y_test)

# Trainer WITHOUT training args (evaluation only)
trainer = Trainer(model=model)

# Predict
preds = trainer.predict(test_dataset)

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

# Metrics
print("==== TEST SET RESULTS ====")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("\nClassification report:\n", classification_report(y_true, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

