from model import create_sliding_window, Model
from config import Config
from data.DataCleaning import DataCleaning
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# ── 基础配置 ──────────────────────────────────────
SAVE_DIR = Config._get_save_directory()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 读取数据 ──────────────────────────────────────
test_data = pd.read_csv(
    r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\test.csv"
)


feature_cols = [
    'cn_interest_rate', 'uk_interest_rate',
    'cn_export', 'cn_import', 'cn_trade_balance',
    'uk_export', 'uk_import',
    'cn_cpi', 'uk_cpi',
    'cn_inflation', 'uk_inflation',
    'cn_gdp', 'uk_gdp',
    'dxy', 'gold_volatility', 'oil_volatility',
    'cn_share_price', 'uk_share_price',
    'price', 'open', 'high', 'low', 'change'
]
target_cols   = ['price']
sequence_length = 45

# ── 滑动窗口切分 ───────────────────────────────────
X_test, Y_test = create_sliding_window(
    test_data, feature_cols, target_cols, sequence_length
)

X_test_t = torch.FloatTensor(X_test)
Y_test_t = torch.FloatTensor(Y_test)

test_loader = DataLoader(
    TensorDataset(X_test_t, Y_test_t),
    batch_size=32,
    shuffle=False
)

# ── 加载模型 ───────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(
    torch.load(
        os.path.join(SAVE_DIR, "CNN_LSTM_model_best.pth"),
        map_location=device  # 防止GPU训练的模型在CPU上加载报错
    )
)
model.eval()

# ── 预测 ───────────────────────────────────────────
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        pred    = model(X_batch)
        predictions.append(pred.cpu().numpy())
        actuals.append(Y_batch.numpy())

predictions = np.concatenate(predictions, axis=0)  # (N, 1)
actuals = np.concatenate(actuals, axis=0)  # (N, 1)

# ── 反归一化 ───────────────────────────────────────
cleaner = DataCleaning()
predictions_real = cleaner.DenormalizeResult(predictions)
actuals_real = cleaner.DenormalizeResult(actuals)

# ── 评估指标 ───────────────────────────────────────
mae = np.mean(np.abs(predictions_real - actuals_real))
rmse = np.sqrt(np.mean((predictions_real - actuals_real) ** 2))

print(f"MAE  (mean absolute error): {mae:.4f}")
print(f"RMSE (root mean squared error)  : {rmse:.4f}")

# ── 画图 ───────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.plot(actuals_real,     label="actual", linewidth=1)
plt.plot(predictions_real, label="predicted", linewidth=1)
plt.xlabel("day")
plt.ylabel("exchange rate (Price)")
plt.title("prediction vs actual")
plt.legend()
plt.tight_layout()
plt.show()