import os
import pandas as pd
from pandas import read_csv

from config import Config

class DataCleaning:
    def __init__(self):
        self.dataset = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\data\train_dataset.csv")
    def normalize(self, df, col):
        max_val = float(self.dataset[col].max())
        min_val = float(self.dataset[col].min())
        df[col] = (df[col] - min_val) / (max_val - min_val)
        return df
    def DenormalizeResult(self, val: int):
        max_val = float(self.dataset["price"].max())
        min_val = float(self.dataset["price"].min())
        val = val*(max_val - min_val) + min_val
        return val


# # 数据标准化：CNY_GBP Historical Data.csv
dataset_location = Config._get_dataset_location()
# file = os.path.join(dataset_location, r"CNY_GBP_Historical_Data.csv")
# df = pd.read_csv(file)
# index = 0
# for date in df["Date"]:
#     month = int(date[:2])
#     day = int(date[3:5])
#     year = int(date[6:])
#     date = f"{year}/{month}/{day}"
#     df.loc[index, "Date"] = date
#     index += 1
#
# df.to_csv(f"{dataset_location}/exchange_rate_data.csv", index=False)
#
# # 合并features与exchange_rate_data
# features_file = os.path.join(dataset_location, r"features.csv")
# exchange_rate_file = os.path.join(dataset_location, r"exchange_rate_data.csv")
# features_df = pd.read_csv(features_file)
# exchange_rate_df = pd.read_csv(exchange_rate_file)
#
# dataset = pd.merge(features_df, exchange_rate_df, on='date', how='left')
# dataset = dataset.ffill()
#
# # 删除change列的百分号
# dataset["change"] = dataset["change"].str[:-1]
# 等比缩小cn_export, uk_export, cn_trade_balance, cn_import, uk_import, cn_gdp, uk_gdp
# dataset["cn_export"] = dataset["cn_export"] / 1e10
# dataset["cn_import"] = dataset["cn_import"] / 1e10
# dataset["uk_export"] = dataset["uk_export"] / 1e10
# dataset["uk_import"] = dataset["uk_import"] / 1e10
# dataset["cn_trade_balance"] = dataset["cn_trade_balance"] / 1e10
# dataset["cn_gdp"] = dataset["cn_gdp"] / 1e12
# dataset["uk_gdp"] = dataset["uk_gdp"] / 1e12
#
#
# # dataset.to_csv(os.path.join(dataset_location, r"dataset1.csv"), index=False)
#
# dataset.to_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\dataset1.csv")
#
# train_df = dataset.iloc[:2744]
# val_df = dataset.iloc[2744:3044]
# test_df = dataset.iloc[3044:]
#
# feature_cols = [
#     'cn_interest_rate', 'uk_interest_rate',
#     'cn_export', 'cn_import', 'cn_trade_balance',
#     'uk_export', 'uk_import',
#     'cn_cpi', 'uk_cpi',
#     'cn_inflation', 'uk_inflation',
#     'cn_gdp', 'uk_gdp',
#     'dxy', 'gold_volatility', 'oil_volatility',
#     'cn_share_price', 'uk_share_price',
#     'price', 'open', 'high', 'low', 'change'
# ]
#
# # 查看每列有多少个 NaN
# print(dataset.isnull().sum())
# # ① 只用训练集计算 min 和 max
# col_min = train_df[feature_cols].min()  # 每列的最小值
# col_max = train_df[feature_cols].max()  # 每列的最大值
# # ② 用同一套 min/max 归一化三个数据集
#
# train_scaled = (train_df[feature_cols] - col_min) / (col_max - col_min)
# val_scaled   = (val_df[feature_cols]   - col_min) / (col_max - col_min)
# test_scaled  = (test_df[feature_cols]  - col_min) / (col_max - col_min)
#
#
# train_scaled.to_csv(os.path.join(dataset_location, 'train.csv'))
# val_scaled.to_csv(os.path.join(dataset_location, 'val.csv'))
# test_scaled.to_csv(os.path.join(dataset_location, 'test.csv'))

