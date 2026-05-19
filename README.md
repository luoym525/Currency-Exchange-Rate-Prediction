# CNY_GBP_exchange_rate_prediction

A deep learning model for predicting the Chinese Yuan (CNY) to British Pound (GBP) exchange rate using a CNN + LSTM hybrid architecture.


## Dataset

The model uses 23 input features to predict the next day's exchange rate price.

**Macroeconomic Features (18):**

- cn_interest_rate, uk_interest_rate
- cn_export, cn_import, cn_trade_balance
- uk_export, uk_import
- cn_cpi, uk_cpi
- cn_inflation, uk_inflation
- cn_gdp, uk_gdp
- dxy, gold_volatility, oil_volatility
- cn_share_price, uk_share_price

**Historical Price Features (5):**

- price, open, high, low, change

**Target Variable:**

- price (next day's CNY/GBP exchange rate)

**Total samples:** 3744 daily records

------
## Model Architecture

1. Input (batch = 45, 23)
2. Conv1D (kernel = 3)
3. ReLU
4. Conv1D (kernal = 5)
5. Dropout(0.3)
6. LSTM * 2 layers (hidden = 128)
7. Take last timestep
8. Linear (128 to 64) + ReLU
9. Linear (64 to 1)
10. Output: predicted exchange rate

---
## Data Preprocessing

1. **Missing value handling:** Forward fill (ffill) + backward fill (bfill)
2. **Normalization:** Min-Max normalization using training set statistics only
3. **Sliding window:** Sequence length = 45 days
4. **Data split (chronological order):**
    - Training set: first 2744 samples
    - Validation set: next 300 samples
    - Test set: last 700 samples
```
Data is split in chronological order。
shuffling is disabled to prevent data leakage
```

___
## Training Config

| **Parameter**           | **Value**                   |
| ----------------------- | --------------------------- |
| Optimizer               | Adam                        |
| Learing rate            | 0.0001                      |
| Loss function           | HuberLoss(delta = 0.01)     |
| Batch size              | 32                          |
| Mac epochs              | 100                         |
| Early stopping patience | 20                          |
| LR scheduler            | CosAnnealingLR (T_max = 50) |
| Dropout                 | 0.3                         |

---
# How to run
## 1. Install dependencies

```python
pip install pandas numpy matplotlib fredapi
```

Then install torch and CUDA based on your own GPU version.

For FredAPI, you have to go to the [FRED website](https://fredaccount.stlouisfed.org/apikeys), sign up and apply for your own API key. **Or you can just use the dataset provided in the dataset folder.**

**Most importantly, go to config.py and write in your own file locations and directory path**

## 2. To train the model

```python
python CNN_LSTM_model/model.py
```

## 3. To evaluate on test set
```python
python CNN_LSTM_model/test.py
```
---
# Results

|Metric|Value|
|---|---|
|MAE (Mean Absolute Error)|0.0095|
|RMSE (Root Mean Squared Error)|0.0098|

The graph for results on test dataset can be found at prediction_to_actual.png


```
Note: Values above are in normalized scale. 
Multiply by (price_max - price_min) to get real exchange rate error.
```
---
## Limitations

- The model performs well on normal market fluctuations
- Extreme events (e.g. Brexit, COVID-19) cause sudden large drops that the model cannot fully predict — this is a known limitation of all historical-data-based forecasting models