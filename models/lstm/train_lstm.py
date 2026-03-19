"""
models/lstm/train_lstm.py
──────────────────────────────────────────────────────────────────────────────
Trains an LSTM model on sequential transaction data.
The LSTM captures temporal patterns that LightGBM cannot — the ORDER and
SEQUENCE of events matters as much as the aggregated signals.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
 
import json
import warnings
from pathlib import Path
 
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
 
warnings.filterwarnings("ignore")
from config.settings import get_settings
 
settings = get_settings()
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)
 
SEQUENCE_LENGTH = 30   # Use 30 most recent transactions
N_FEATURES = 9        # Behavioral feature vector size
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
N_LAYERS = 2
DROPOUT = 0.3
 
 
# ── Dataset ───────────────────────────────────────────────────────────────────
class TransactionSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.labels    = torch.FloatTensor(labels)
 
    def __len__(self) -> int:
        return len(self.labels)
 
    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx]
 
 
# ── Model ─────────────────────────────────────────────────────────────────────
class SentinelLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_out = lstm_out[:, -1, :]
        dropped = self.dropout(last_out)
        logits = self.fc(dropped)
        return self.sigmoid(logits).squeeze(1)
 
 
# ── Build sequences ───────────────────────────────────────────────────────────
FEATURES = [
    "salary_delay_days", "balance_wow_drop_pct", "upi_lending_spike_ratio",
    "utility_payment_latency", "discretionary_contraction", "atm_withdrawal_spike",
    "failed_auto_debit_count", "credit_utilization_delta", "drift_score",
]
 
 
def build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert tabular training data into padded sequences per customer."""
    sequences = []
    labels = []
    for _, cust_df in df.groupby("customer_id"):
        cust_df = cust_df.sort_values("reference_date").reset_index(drop=True)
        feat_array = cust_df[FEATURES].fillna(0).values.astype(np.float32)
        label_array = cust_df["label"].values
        # Slide a window over the sequence
        for i in range(len(feat_array)):
            start = max(0, i - SEQUENCE_LENGTH + 1)
            seq = feat_array[start:i+1]
            # Pad from the left if sequence is shorter than SEQUENCE_LENGTH
            if len(seq) < SEQUENCE_LENGTH:
                pad = np.zeros((SEQUENCE_LENGTH - len(seq), N_FEATURES), dtype=np.float32)
                seq = np.vstack([pad, seq])
            sequences.append(seq[:SEQUENCE_LENGTH])
            labels.append(float(label_array[i]))
    return np.array(sequences), np.array(labels)
 
 
def train(data_path: str = "data/training_features.parquet") -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
 
    df = pd.read_parquet(data_path).dropna(subset=FEATURES + ["label"])
    X_seq, y = build_sequences(df)
    print(f"Sequences: {X_seq.shape}, Labels: {y.shape}")
 
    X_train, X_val, y_train, y_val = train_test_split(X_seq, y, test_size=0.2, stratify=y, random_state=42)
    train_ds = TransactionSequenceDataset(X_train, y_train)
    val_ds   = TransactionSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
 
    with mlflow.start_run(run_name="lstm-v1") as run:
        model = SentinelLSTM(N_FEATURES, HIDDEN_SIZE, N_LAYERS, DROPOUT).to(device)
        pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
 
        mlflow.log_params({
            "hidden_size": HIDDEN_SIZE, "n_layers": N_LAYERS,
            "dropout": DROPOUT, "lr": LEARNING_RATE, "epochs": EPOCHS,
        })
 
        best_val_auc = 0.0
        for epoch in range(EPOCHS):
            # Train
            model.train()
            train_loss = 0.0
            for seqs, lbls in train_loader:
                seqs, lbls = seqs.to(device), lbls.to(device)
                optimizer.zero_grad()
                preds = model(seqs)
                loss = criterion(preds, lbls)
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
 
            # Validate
            model.eval()
            val_probs = []
            val_labels = []
            with torch.no_grad():
                for seqs, lbls in val_loader:
                    seqs = seqs.to(device)
                    probs = model(seqs).cpu().numpy()
                    val_probs.extend(probs)
                    val_labels.extend(lbls.numpy())
            val_auc = roc_auc_score(val_labels, val_probs)
            scheduler.step(1 - val_auc)
 
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), "models/lstm/lstm_best.pt")
 
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
                mlflow.log_metrics({"val_auc": val_auc, "train_loss": train_loss / len(train_loader)}, step=epoch)
 
        print(f"\nBest Val AUC: {best_val_auc:.4f}")
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_artifact("models/lstm/lstm_best.pt", "lstm_model")
 
        # Save model config
        cfg = {"hidden_size": HIDDEN_SIZE, "n_layers": N_LAYERS, "dropout": DROPOUT,
               "n_features": N_FEATURES, "seq_length": SEQUENCE_LENGTH, "features": FEATURES}
        with open("models/lstm/model_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
 
        print(f"LSTM training complete. Run ID: {run.info.run_id}")
        return run.info.run_id
 
 
if __name__ == "__main__":
    train()
