import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import (
    FEATURE_COLS,
    SEQUENCE_LENGTH,
    TARGET_COLS,
    Model,
    build_loaders,
    create_sliding_window,
    denormalize_price,
    load_datasets,
    load_price_scale,
    to_absolute_prediction,
)

SAVE_DIR = Config._get_save_directory()
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "CNN_LSTM_model_best.pth")






def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = load_price_scale()
    predict_delta = scale.get("predict_delta", True)

    test_data = load_datasets()[2]
    X_test, Y_test = create_sliding_window(
        test_data,
        FEATURE_COLS,
        TARGET_COLS,
        SEQUENCE_LENGTH,
        predict_delta=predict_delta,
    )

    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test)),
        batch_size=32,
        shuffle=False,
    )

    model = Model().to(device)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    predictions, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            predictions.append(pred)
            actuals.append(yb.numpy())

    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    if predict_delta:
        predictions = to_absolute_prediction(predictions, X_test)
        actuals = to_absolute_prediction(actuals, X_test)

    predictions_real = denormalize_price(predictions, scale)
    actuals_real = denormalize_price(actuals, scale)

    mae = np.mean(np.abs(predictions_real - actuals_real))
    rmse = np.sqrt(np.mean((predictions_real - actuals_real) ** 2))
    pred_std = float(np.std(predictions_real))

    print(f"MAE  (mean absolute error): {mae:.6f}")
    print(f"RMSE (root mean squared error): {rmse:.6f}")
    print(f"Prediction std (real scale): {pred_std:.6f}")
    if pred_std < 1e-6:
        print("WARNING: predictions are nearly constant — check training.")

    plt.figure(figsize=(12, 5))
    plt.plot(actuals_real, label="actual", linewidth=1)
    plt.plot(predictions_real, label="predicted", linewidth=1)
    plt.xlabel("day")
    plt.ylabel("exchange rate (Price)")
    plt.title("prediction vs actual")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, "prediction_vs_actual.png")
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
