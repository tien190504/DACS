import torch
import numpy as np

from data_loader import load_data, create_enhanced_sequences, split_data, create_data_loaders
from preprocessing import preprocess_data
from model import create_model
from evaluation import comprehensive_evaluation, inverse_close_transform

from config import WINDOW, PREDICTION_HORIZON, DEVICE, MODEL_SAVE_PATH

def main():
    # 1. Load raw CSV data
    df_senti, df_price = load_data()
    print(f"Sentiment data shape: {df_senti.shape}")
    print(f"Price data shape: {df_price.shape}")

    # 2. Preprocess data: compute indicators, merge sentiment, scale
    df_processed, selected_features, scalers = preprocess_data(df_senti, df_price)
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Selected features count: {len(selected_features)}")

    # 3. Create sequences (X, y) for model
    X, y = create_enhanced_sequences(
        df_processed,
        selected_features,
        WINDOW,
        PREDICTION_HORIZON
    )
    print(f"Sequences shapes: X={X.shape}, y={y.shape}")

    # 4. Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test set:  X={X_test.shape}, y={y_test.shape}")

    # 5. Create DataLoader for test set
    _, _, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)
    print("DataLoader cho test đã sẵn sàng.")

    # 6. Initialize model and load weights
    model = create_model(
        feature_dim=len(selected_features),
        window=WINDOW,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        device=DEVICE
    )
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    print("Model đã load weights xong và chuyển sang eval() mode.")

    # 7. Run comprehensive evaluation on test set
    results = comprehensive_evaluation(
        model=model,
        test_loader=test_loader,
        scaler_price=scalers['price'],
        price_cols=scalers['price_cols'],
        train_losses=[],
        val_losses=[],
        device=DEVICE
    )

    # 8. Optional: example inference for first batch
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            preds = model(X_batch)
            preds_real = inverse_close_transform(preds[:5].cpu().numpy().reshape(-1, 1),
                                                 scalers['price'],
                                                 scalers['price_cols'])
            true_real = inverse_close_transform(y_batch[:5].cpu().numpy().reshape(-1, 1),
                                                scalers['price'],
                                                scalers['price_cols'])
            for i in range(5):
                print(f"Sample {i+1}: Predicted Close=${preds_real[i]:.2f} | Actual Close=${true_real[i]:.2f}")
            break

if __name__ == "__main__":
    main()