import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, roc_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import os
import pandas_ta as ta
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import warnings

warnings.filterwarnings('ignore')

# ====================== ENHANCED CONFIG ======================
# --- File paths & data ---
from src.config import S_AND_P_500_NEWS_DATA, S_AND_P_500_PRICES_DATA

SENTI_FILENAME = 'labeled_News_dataset_have_date.csv'
PRICE_FILENAME = 'sp500.csv'

SENTI_FILE = os.path.join(S_AND_P_500_NEWS_DATA, SENTI_FILENAME)
PRICE_FILE = os.path.join(S_AND_P_500_PRICES_DATA, PRICE_FILENAME)

# --- Enhanced Features ---
PRICE_FEATURES = [
    'S&P500', 'Open', 'High', 'Low', 'Volume',
    'RSI_14', 'RSI_7', 'RSI_21',
    'EMA_10', 'EMA_20', 'EMA_50', 'SMA_10', 'SMA_20', 'SMA_50',
    'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_percent',
    'ATR', 'ADX', 'CCI', 'STOCH_k', 'STOCH_d',
    'OBV', 'MFI', 'WILLR', 'ROC'
]

EMOTION_COLS = ['label', 'sad', 'joy', 'love', 'anger', 'fear', 'surprise']

# --- Enhanced model parameters ---
WINDOW = 30  # Increased window size
PREDICTION_HORIZON = 1  # Can be adjusted for multi-step prediction
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 5e-5
D_MODEL = 128
N_HEAD = 16
N_LAYERS = 6
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Enhanced Early Stopping ---
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_DELTA = 1e-6
MODEL_SAVE_PATH = 'enhanced_transformer_model.pt'

# ================== 1. ENHANCED DATA LOADING & PREPROCESSING ===================
print("Loading and preprocessing data...")

df_senti = pd.read_csv(SENTI_FILE, encoding='latin1')
df_price = pd.read_csv(PRICE_FILE)

# Enhanced technical indicators
def add_enhanced_indicators(df):
    # RSI variations
    df['RSI_7'] = ta.rsi(df['Close'], length=7)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_21'] = ta.rsi(df['Close'], length=21)

    # Moving averages
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)

    # MACD
    macd = ta.macd(df['Close'])
    for col_src, col_dst in zip(['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9'],
                                ['MACD', 'MACD_signal', 'MACD_hist']):
        df[col_dst] = macd[col_src] if col_src in macd else 0

    # Bollinger Bands
    bb = ta.bbands(df['Close'])
    df['BB_upper'] = bb['BBU_5_2.0']
    df['BB_middle'] = bb['BBM_5_2.0']
    df['BB_lower'] = bb['BBL_5_2.0']
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Other indicators
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'])['ADX_14']
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])

    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df['STOCH_k'] = stoch['STOCHk_14_3_3']
    df['STOCH_d'] = stoch['STOCHd_14_3_3']

    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])
    df['ROC'] = ta.roc(df['Close'])

    # Price-based features
    df['price_change'] = df['Close'].pct_change()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['volume_sma_ratio'] = df['Volume'] / ta.sma(df['Volume'], length=20)

    return df

df_price = add_enhanced_indicators(df_price)

# Enhanced sentiment processing
df_senti['DATE'] = pd.to_datetime(df_senti['DATE'], errors='coerce')
df_senti = df_senti.dropna(subset=['DATE'])
df_price['DATE'] = pd.to_datetime(df_price['DATE'], errors='coerce')
df_price = df_price.dropna(subset=['DATE'])

# Process emotions with additional features
for col in EMOTION_COLS:
    df_senti[col] = pd.to_numeric(df_senti[col], errors='coerce')
df_senti = df_senti.dropna(subset=EMOTION_COLS)

# Enhanced sentiment aggregation
df_daily_senti = df_senti.groupby('DATE').agg({
    **{col: ['mean', 'std', 'max', 'min'] for col in EMOTION_COLS}
}).reset_index()
df_daily_senti.columns = ['DATE'] + [f"{col[0]}_{col[1]}" for col in df_daily_senti.columns[1:]]

# Merge data
df = pd.merge(df_price, df_daily_senti, on='DATE', how='inner')
df = df.sort_values('DATE').reset_index(drop=True)
df = df.fillna(method='ffill').fillna(method='bfill')

# Select final features
all_features = []
for col in df.columns:
    if col not in ['DATE', 'Adj Close'] and df[col].dtype in ['float64', 'int64']:
        all_features.append(col)

print(f"Total features: {len(all_features)}")

# Feature scaling with multiple scalers
scaler_price = RobustScaler()
scaler_volume = StandardScaler()
scaler_indicators = MinMaxScaler()
scaler_sentiment = StandardScaler()

price_cols = ['Open', 'High', 'Low', 'Close']
volume_cols = ['Volume', 'OBV']
indicator_cols = [col for col in all_features if any(
    x in col for x in ['RSI', 'MACD', 'BB', 'ATR', 'ADX', 'CCI', 'STOCH', 'MFI', 'WILLR', 'ROC', 'TSI', 'EMA', 'SMA'])]
sentiment_cols = [col for col in all_features if any(x in col for x in EMOTION_COLS)]

df[price_cols] = scaler_price.fit_transform(df[price_cols])
df[volume_cols] = scaler_volume.fit_transform(df[volume_cols])
df[indicator_cols] = scaler_indicators.fit_transform(df[indicator_cols])
if sentiment_cols:
    df[sentiment_cols] = scaler_sentiment.fit_transform(df[sentiment_cols])

# Feature selection using correlation and importance
correlation_matrix = df[all_features].corr()
high_corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)
selected_features = [col for col in all_features if col not in high_corr_features]
print(f"Selected features after correlation filtering: {len(selected_features)}")

# ================== 2. ENHANCED SLIDING WINDOW WITH MULTI-STEP ===================
def create_enhanced_sequences(df, features, window_size, prediction_horizon=1):
    X, y = [], []
    for i in range(len(df) - window_size - prediction_horizon + 1):
        sequence = df[features].iloc[i:i + window_size].values
        target = df['Close'].iloc[i + window_size:i + window_size + prediction_horizon].values
        if prediction_horizon == 1:
            target = target[0]
        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)

X, y = create_enhanced_sequences(df, selected_features, WINDOW, PREDICTION_HORIZON)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ================== 3. ENHANCED TRAIN/VAL/TEST SPLIT ===================
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create weighted sampling for imbalanced data
def create_weighted_sampler(y_data, window_size=20):
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

train_sampler = create_weighted_sampler(y_train.numpy())

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE,
                          sampler=train_sampler if train_sampler else None,
                          shuffle=train_sampler is None)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

# ================== 4. ENHANCED MODEL ARCHITECTURE ===================
class EnhancedTimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, window, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.window = window
        self.feature_dim = feature_dim

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        self.pos_encoding = nn.Parameter(torch.randn(window, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, d_model))

        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )

        self.direction_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # up/down classification
        )

    def forward(self, x, return_direction=False):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        transformer_out = self.transformer(x)
        query = self.pool_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_out, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_out = pooled_out.squeeze(1)
        price_pred = self.output_layers(pooled_out).squeeze(-1)
        if return_direction:
            direction_logits = self.direction_classifier(pooled_out)
            return price_pred, direction_logits
        return price_pred

model = EnhancedTimeSeriesTransformer(
    feature_dim=len(selected_features),
    window=WINDOW,
    d_model=D_MODEL,
    nhead=N_HEAD,
    num_layers=N_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ================== 5. ENHANCED TRAINING WITH MULTI-TASK LEARNING ===================
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
regression_loss_fn = nn.SmoothL1Loss()
classification_loss_fn = nn.CrossEntropyLoss()

def multi_task_loss(price_pred, price_true, direction_logits=None, direction_true=None, alpha=0.7):
    reg_loss = regression_loss_fn(price_pred, price_true)
    if direction_logits is not None and direction_true is not None:
        cls_loss = classification_loss_fn(direction_logits, direction_true)
        return alpha * reg_loss + (1 - alpha) * cls_loss, reg_loss, cls_loss
    return reg_loss, reg_loss, torch.tensor(0.0)

def get_direction_labels(y_batch):
    directions = torch.zeros(len(y_batch), dtype=torch.long, device=y_batch.device)
    return directions

def evaluate_model(model, data_loader, device, return_predictions=False):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = regression_loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
            if return_predictions:
                predictions.append(pred.cpu().numpy())
                targets.append(yb.cpu().numpy())
    avg_loss = total_loss / len(data_loader.dataset)
    if return_predictions:
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        return avg_loss, predictions, targets
    return avg_loss

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

def train_main():
    print("Starting enhanced training...")
    global best_val_loss, patience_counter, train_losses, val_losses
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            price_pred, direction_logits = model(xb, return_direction=True)
            direction_labels = get_direction_labels(yb)
            loss, reg_loss, cls_loss = multi_task_loss(
                price_pred, yb, direction_logits, direction_labels
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total_reg_loss += reg_loss.item() * xb.size(0)
            total_cls_loss += cls_loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Reg Loss: {total_reg_loss / len(train_loader.dataset):.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        if val_loss + EARLY_STOPPING_DELTA < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}. Best Val Loss: {best_val_loss:.6f}")
                break

    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    # ================== 6. ENHANCED EVALUATION & METRICS ===================
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, y_pred, y_true = evaluate_model(model, test_loader, DEVICE, return_predictions=True)

    def inverse_close_transform(val):
        close_idx = price_cols.index('Close')
        return scaler_price.inverse_transform(
            np.column_stack(
                [np.zeros((len(val), close_idx)), val, np.zeros((len(val), len(price_cols) - close_idx - 1))])
        )[:, close_idx]

    y_true_real = inverse_close_transform(y_true.reshape(-1, 1))
    y_pred_real = inverse_close_transform(y_pred.reshape(-1, 1))

    # Regression metrics
    mse = mean_squared_error(y_true_real, y_pred_real)
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100

    print(f"\n=== REGRESSION METRICS ===")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAPE: {mape:.2f}%")

    def calc_direction_with_threshold(arr, threshold=0.001):
        changes = np.diff(arr)
        return (changes > threshold).astype(int)

    y_true_cls = calc_direction_with_threshold(y_true_real, threshold=np.std(y_true_real) * 0.1)
    y_pred_cls = calc_direction_with_threshold(y_pred_real, threshold=np.std(y_pred_real) * 0.1)
    min_len = min(len(y_true_cls), len(y_pred_cls))
    y_true_cls = y_true_cls[:min_len]
    y_pred_cls = y_pred_cls[:min_len]

    # Classification metrics
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls, average='weighted')
    try:
        auc_roc = roc_auc_score(y_true_cls, y_pred_cls)
    except ValueError:
        auc_roc = 0.5
    try:
        precision, recall, _ = precision_recall_curve(y_true_cls, y_pred_cls)
        pr_auc = auc(recall, precision)
    except ValueError:
        pr_auc = 0.5

    print(f"\n=== CLASSIFICATION METRICS (Direction Prediction) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    def calculate_trading_metrics(y_true, y_pred, transaction_cost=0.001):
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        positions = np.where(pred_returns > 0, 1, -1)
        transaction_costs = transaction_cost * np.abs(np.diff(positions))
        transaction_costs = np.insert(transaction_costs, 0, 0)
        strategy_returns = positions * true_returns - transaction_costs
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(
            strategy_returns) > 0 else 0
        max_drawdown = np.min(np.cumsum(strategy_returns))
        return total_return, sharpe_ratio, max_drawdown

    total_return, sharpe_ratio, max_drawdown = calculate_trading_metrics(y_true_real, y_pred_real)

    print(f"\n=== TRADING SIMULATION METRICS ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # ================== 7. ENHANCED VISUALIZATION ===================
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].plot(y_true_real[-200:], label='True Close', alpha=0.8)
    axes[0, 0].plot(y_pred_real[-200:], label='Predicted Close', alpha=0.8)
    axes[0, 0].set_title('S&P 500 Close Price Prediction (Last 200 days)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    residuals = y_true_real - y_pred_real
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Prediction Error Distribution')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    if len(np.unique(y_true_cls)) > 1:
        fpr, tpr, _ = roc_curve(y_true_cls, y_pred_cls)
        axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(train_losses, label='Training Loss', alpha=0.8)
    axes[1, 1].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[1, 1].set_title('Training History')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'=' * 50}")
    print(f"ENHANCED MODEL PERFORMANCE SUMMARY")
    print(f"{'=' * 50}")
    print(f"Direction Accuracy: {acc:.4f} (Target: 0.8000)")
    print(f"Price RMSE: {rmse:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"{'=' * 50}")

    if acc >= 0.8:
        print("🎉 TARGET ACCURACY ACHIEVED!")

if __name__ == '__main__':
    train_main()