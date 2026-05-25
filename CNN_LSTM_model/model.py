import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import Config

SAVE_DIR = Config._get_save_directory()
SCALE_PATH = os.path.join(SAVE_DIR, "price_scale.json")

FEATURE_COLS = [
    "cn_interest_rate", "uk_interest_rate",
    "cn_export", "cn_import", "cn_trade_balance",
    "uk_export", "uk_import",
    "cn_cpi", "uk_cpi",
    "cn_inflation", "uk_inflation",
    "cn_gdp", "uk_gdp",
    "dxy", "gold_volatility", "oil_volatility",
    "cn_share_price", "uk_share_price",
    "price", "open", "high", "low", "change",
]
TARGET_COLS = ["price"]
SEQUENCE_LENGTH = 45
PRICE_IDX = FEATURE_COLS.index("price")
N_FEATURES = len(FEATURE_COLS)


def create_sliding_window(data, feature_cols, target_cols, sequence_length, predict_delta=False):
    X, Y = [], []
    feature = data[feature_cols].values
    target = data[target_cols].values
    for idx in range(sequence_length, len(data)):
        window = feature[idx - sequence_length:idx]
        X.append(window)
        y = target[idx]
        if predict_delta:
            y = y - window[-1, PRICE_IDX : PRICE_IDX + 1]
        Y.append(y)
    return np.array(X), np.array(Y)


def load_datasets(data_dir="../dataset"):
    train_data = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_data = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_data, val_data, test_data


def build_loaders(batch_size=32, predict_delta=True, shuffle_train=True):
    train_data, val_data, test_data = load_datasets()
    X_train, Y_train = create_sliding_window(
        train_data, FEATURE_COLS, TARGET_COLS, SEQUENCE_LENGTH, predict_delta
    )
    X_val, Y_val = create_sliding_window(
        val_data, FEATURE_COLS, TARGET_COLS, SEQUENCE_LENGTH, predict_delta
    )
    X_test, Y_test = create_sliding_window(
        test_data, FEATURE_COLS, TARGET_COLS, SEQUENCE_LENGTH, predict_delta
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train)),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, (X_test, Y_test)


def save_price_scale():
    """训练集原始 price 的 min/max，用于反归一化（与 dataset 生成逻辑一致）。"""
    full = pd.read_csv(os.path.join(os.path.dirname(SAVE_DIR), "data", "prev_dataset.csv"))
    train_raw = full.iloc[:2744]
    scale = {
        "price_min": float(train_raw["price"].min()),
        "price_max": float(train_raw["price"].max()),
        "predict_delta": True,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SCALE_PATH, "w", encoding="utf-8") as f:
        json.dump(scale, f, indent=2)
    return scale


def load_price_scale():
    with open(SCALE_PATH, encoding="utf-8") as f:
        return json.load(f)


def denormalize_price(norm_values, scale=None):
    scale = scale or load_price_scale()
    arr = np.asarray(norm_values, dtype=np.float64)
    return arr * (scale["price_max"] - scale["price_min"]) + scale["price_min"]


def to_absolute_prediction(norm_delta_pred, windows):
    """残差预测 + 窗口末 price -> 归一化绝对 price。"""
    last_price = windows[:, -1, PRICE_IDX : PRICE_IDX + 1]
    return np.asarray(norm_delta_pred) + last_price


class Model(nn.Module):
    """
    CNN 提取局部时序模式 -> 2 层 LSTM -> 全连接头。
    默认预测相对上一日的归一化价差（残差），更利于非平稳汇率序列。
    """

    def __init__(
        self,
        n_features=N_FEATURES,
        cnn_channels=32,
        lstm_hidden=96,
        lstm_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm=1.0):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        total += criterion(model(xb), yb).item()
    return total / len(loader)


if __name__ == "__main__":
    save_price_scale()
    train_loader, val_loader, test_loader, _ = build_loaders(
        batch_size=32, predict_delta=True, shuffle_train=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    num_epochs = 100
    patience = 20
    best_val_loss = float("inf")
    patience_counter = 0
    train_loss_array, val_loss_array = [], []

    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_loss_array.append(train_loss)
        val_loss_array.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train: {train_loss:.6f} | val: {val_loss:.6f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, f"CNN_LSTM_model_{epoch + 1}.pth"),
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR, "CNN_LSTM_model_best.pth"),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch + 1}")
                break

    plt.plot(train_loss_array, label="train loss")
    plt.plot(val_loss_array, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("train & validation loss")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_loss.png"))
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Saved best model to {os.path.join(SAVE_DIR, 'CNN_LSTM_model_best.pth')}")
