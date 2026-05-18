import pandas as pd

file = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\dataset.csv")

cn_interest_rate = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\cn_interest_rate.csv")
uk_interest_rate = pd.read_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\uk_interest_rate.csv")

dataset = pd.merge(file, cn_interest_rate, on='date', how='left')
dataset = pd.merge(dataset, uk_interest_rate, on='date', how='left')
dataset = dataset.ffill()
print(dataset.head())

dataset.to_csv(r"D:\code\CNY_GBP_exchange_rate_prediction\dataset\dataset1.csv")