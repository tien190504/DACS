import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from config import SENTI_FILE, PRICE_FILE, EMOTION_COLS, BATCH_SIZE
import warnings

warnings.filterwarnings('ignore')


def load_data():
    """Load sentiment and price data from CSV files."""
    print("Loading data...")
    df_senti = pd.read_csv(SENTI_FILE, encoding='latin1')
    df_price = pd.read_csv(PRICE_FILE)
    return df_senti, df_price


def create_enhanced_sequences(df, features, window_size, prediction_horizon=1):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(df) - window_size - prediction_horizon + 1):
        sequence = df[features].iloc[i:i + window_size].values
        target = df['Close'].iloc[i + window_size:i + window_size + prediction_horizon].values
        if prediction_horizon == 1:
            target = target[0]
        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)


def create_weighted_sampler(y_data, window_size=20):
    """Create weighted sampler for imbalanced data."""
    price_changes = []
    for i in range(len(y_data) - 1):
        change = 1 if y_data[i + 1] > y_data[i] else 0
        price_changes.append(change)

    if len(price_changes) == 0:
        return None

    unique, counts = np.unique(price_changes, return_counts=True)
    class_weights = 1.0 / counts
    weights = [class_weights[change] for change in price_changes]
    weights.extend([weights[-1]] * (len(y_data) - len(weights)))

    return WeightedRandomSampler(weights, len(weights))


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    """Create data loaders for training, validation, and testing."""
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create weighted sampling for imbalanced data
    train_sampler = create_weighted_sampler(y_train.numpy())

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        sampler=train_sampler if train_sampler else None,
        shuffle=train_sampler is None
    )

    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets."""
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)

    X_train = X[:train_size]
    X_val = X[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]

    y_train = y[:train_size]
    y_val = y[train_size:train_size + val_size]
    y_test = y[train_size + val_size:]

    return X_train, X_val, X_test, y_train, y_val, y_test