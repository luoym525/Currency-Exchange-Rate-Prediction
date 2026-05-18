import pandas as pd
from fredapi import Fred
import os

from config import Config

FRED_API_KEY = Config._get_Fred_api()
fred = Fred(api_key=FRED_API_KEY)
dataset_location = Config._get_dataset_location()
START = "2016-01-01"
END = "2026-03-31"

# 下载所有 FRED 宏观经济数据
fred_series = {
    # 利率
    "cn_interest_rate": "INTDSRCNM193N",  # 中国贴现率（月度）
    "uk_interest_rate": "BOERUKM",  # 英国基准利率（月度）
    # 进出口 & 贸易差额
    "cn_exports": "XTEXVA01CNM667S",  # 中国出口（月度）
    "cn_imports": "XTIMVA01CNM667S",  # 中国进口（月度）
    "cn_trade_balance": "XTNTVA01CNM667S",  # 中国贸易差额（月度）
    "uk_exports": "XTEXVA01GBM664S",  # 英国出口（月度）
    "uk_imports": "XTIMVA01GBM664S",  # 英国进口（月度）
    "uk_trade_balance": "XTNTVA01GBM664N",  # 英国贸易差额（月度）
    # CPI（月度）
    "cn_cpi": "CHNCPIALLMINMEI",  # 中国 CPI
    "uk_cpi": "GBRCPIALLMINMEI",  # 英国 CPI
    # 通胀率（年度，训练时需填充为月度）
    "cn_inflation": "FPCPITOTLZGCHN",  # 中国通胀率
    "uk_inflation": "FPCPITOTLZGGBR",  # 英国通胀率
    # GDP（年度，训练时需填充为月度）
    "cn_gdp": "MKTGDPCNA646NWDB",  # 中国 GDP
    "uk_gdp": "MKTGDPGBA646NWDB",  # 英国 GDP
    # 金融指数
    "dxy": "DTWEXBGS",  # 美元指数
    "oil_wti": "DCOILWTICO",  # 原油价格（日度）
    "gold_volatility":  "GVZCLS",              # 黄金 ETF 波动率指数（日度）
    "oil_volatility":   "OVXCLS",              # 原油 ETF 波动率指数（日度）
    "cn_share_price":   "SPASTT01CNM661N",     # 中国股市价格指数（月度）
    "uk_share_price":   "SPASTT01GBM661N",     # 英国股市价格指数（月度）
    }
macro_data = {}
for name, series_id in fred_series.items():
    try:
        series = fred.get_series(series_id, observation_start = START, observation_end = END)
        macro_data[name] = series
        print(f"✅ Successfully downloaded: {name:25s} | {len(series):4d} 条 | 最新: {series.index[-1].date()}")
    except Exception as e:
        print(f"❌ Failed {name:25s} | 失败: {e}")
# 合并成一张表，按月对齐
macro_df = pd.DataFrame(macro_data)
macro_df.index = pd.to_datetime(macro_df.index)
macro_df = macro_df.resample("D").last()
macro_df = macro_df.ffill()
macro_df.to_csv(os.path.join(dataset_location, "features1.csv"), index = True)




# # 下载市场数据(yfinance)
#
# START = "2016-01-01"
# END = "2026-03-31"
# import yfinance as yf
# df = yf.download("CNYGBP=X", start = START, end = END, auto_adjust=True) # max = 全部历史
# df.to_csv("CNY_GBP_history.csv")
# print(df.head())
# print("✅ 完成！")
# market_ticker = {
#     "cny_gbp":    "CNYGBP=X",   # x = 人民币->英镑汇率
#     "ftse100":    "^FTSE",       # 英国股市
#     "csi300":     "000300.SS",   # 沪深 300
#     "dxy":        "DX-Y.NYB",    # 美元指数
#     "oil_wti":    "CL=F",        # 原油价格
#     "gold":       "GC=F",        # 黄金价格
# }
# market_data = {}
# for name, ticker in market_ticker.items():
#     try:
#         df = yf.download(ticker, start = START, end = END, auto_adjust=True)
#         series = df["Close"].squeeze()
#         series.index = pd.to_datetime(series.index)
#
#         market_data[name] = series
#         print(f"✅ {name:12s} | {len(series):4d} 条日度数据")
#     except Exception as e:
#         print(f"❌ {name:12s} | 失败: {e}")
#     time.sleep(10)
# market_df = pd.DataFrame(market_data)
#
# full_df = pd.concat([macro_df, market_df], axis=1)
# full_df = full_df.ffill()
# full_df.dropna(subset=["cny_gbp"], inplace=True)
# full_df.to_csv(os.path.join(dataset_location, "features.csv"), index = True)

