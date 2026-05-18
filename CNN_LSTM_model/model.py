import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

train_data = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\train.csv")
test_data = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\test.csv")
val_data = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\val.csv")

feature_cols = [
    # 宏观经济特征（18列）
    'cn_interest_rate', 'uk_interest_rate',
    'cn_export', 'cn_import', 'cn_trade_balance',
    'uk_export', 'uk_import',
    'cn_cpi', 'uk_cpi',
    'cn_inflation', 'uk_inflation',
    'cn_gdp', 'uk_gdp',
    'dxy', 'gold_volatility', 'oil_volatility',
    'cn_share_price', 'uk_share_price',

    # 历史价格特征（5列）
    'price', 'open', 'high', 'low', 'change'
]

target_cols = ['price']

sequence_length = 60

def create_sliding_window(data, feature_cols, target_cols, sequence_length):
    """
    data        : 归一化后的 DataFrame
    feature_cols: 输入特征列名列表
    target_cols : 目标列名列表
    seq_len     : 窗口大小（用过去多少天）
    """
    X = []
    Y = []
    feature = data[feature_cols].values
    target = data[target_cols].values
    for idx in range(sequence_length, len(data)):
        X.append(feature[idx - sequence_length:idx])
        Y.append(target[idx])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# 对训练集切分
X_train, Y_train = create_sliding_window(
    train_data, feature_cols, target_cols, sequence_length)
# 对验证集切分
X_val, Y_val = create_sliding_window(
    val_data, feature_cols, target_cols, sequence_length)
# 对测试集切分
X_test, Y_test = create_sliding_window(
    test_data, feature_cols, target_cols, sequence_length)

# print(f"训练集 X: {X_train.shape}")  # (2684, 60, 23)
# print(f"训练集 Y: {Y_train.shape}")  # (2684, 1)
# print(f"验证集 X: {X_val.shape}")    # (240, 60, 23)
# print(f"验证集 Y: {Y_val.shape}")    # (240, 1)
# print(f"测试集 X: {X_test.shape}")   # (639, 60, 23)
# print(f"测试集 Y: {Y_test.shape}")   # (639, 1)

X_train_t = torch.FloatTensor(X_train)
Y_train_t = torch.FloatTensor(Y_train)
X_val_t   = torch.FloatTensor(X_val)
Y_val_t   = torch.FloatTensor(Y_val)
X_test_t  = torch.FloatTensor(X_test)
Y_test_t  = torch.FloatTensor(Y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=32,
    shuffle=False
)
val_loader = DataLoader(
    TensorDataset(X_val_t, Y_val_t),
    batch_size=32,
    shuffle=False
)
test_loader = DataLoader(
    TensorDataset(X_test_t, Y_test_t),
    batch_size=32,
    shuffle=False
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积部分
        self.cnn = nn.Sequential(
        # 第一层卷积，捕捉3天短期模式
        nn.Conv1d(
            in_channels=23,
            out_channels=64,
            kernel_size=3,
            padding=1
        ),
        nn.ReLU(),
        # 第二层卷积，捕捉5天短期模式
        nn.Conv1d(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            padding=2
        ),
        nn.ReLU(),
        nn.Dropout(0.2)
        )
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size = 64,
            hidden_size = 128,
            num_layers = 2,
            batch_first = True,
            dropout = 0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, x):
        "x: (batch = 32, sequence_length = 60, features = 23)"
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # 只取最后一个时间步
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# model = Model()
# print(model)
# test_input = torch.randn(32, 60, 23)
# test_output = model(test_input)
# print(f"\n输入形状：{test_input.shape}")   # (32, 60, 23)
# print(f"输出形状：{test_output.shape}")   # (32, 1)