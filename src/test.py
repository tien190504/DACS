import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from config import S_AND_P_500_NEWS_DATA, S_AND_P_500_PRICES_DATA
import os

# ======== CONFIG =========
MODEL_PATH = 'enhanced_transformer_model1.pt'
TEST_DATA_PATH = 'sp500.csv'   # Đường dẫn file test, thay nếu cần
SENTI_PATH = 'labeled_News_dataset_have_date.csv'
WINDOW = 30

# ---- Các scaler cần đúng với lúc train! ----
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas_ta as ta

# ====== LOAD DATA & FEATURE ENGINEERING (giống hệt file train) ======
def add_enhanced_indicators(df):
    df['RSI_7'] = ta.rsi(df['Close'], length=7)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    df['RSI_21'] = ta.rsi(df['Close'], length=21)
    df['EMA_10'] = ta.ema(df['Close'], length=10)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['SMA_10'] = ta.sma(df['Close'], length=10)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    bb = ta.bbands(df['Close'])
    df['BB_upper'] = bb['BBU_5_2.0']
    df['BB_middle'] = bb['BBM_5_2.0']
    df['BB_lower'] = bb['BBL_5_2.0']
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
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
    df['price_change'] = df['Close'].pct_change()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['volume_sma_ratio'] = df['Volume'] / ta.sma(df['Volume'], length=20)
    return df

# === LOAD DATA ===
input_prices = os.path.join(S_AND_P_500_PRICES_DATA, TEST_DATA_PATH)
df_price = pd.read_csv(input_prices)
df_price = add_enhanced_indicators(df_price)
df_price['DATE'] = pd.to_datetime(df_price['DATE'], errors='coerce')
df_price = df_price.dropna(subset=['DATE'])

input_senti = os.path.join(S_AND_P_500_NEWS_DATA, SENTI_PATH)
df_senti = pd.read_csv(input_senti, encoding='latin1')
df_senti['DATE'] = pd.to_datetime(df_senti['DATE'], errors='coerce')
EMOTION_COLS = ['label', 'sad', 'joy', 'love', 'anger', 'fear', 'surprise']
for col in EMOTION_COLS:
    df_senti[col] = pd.to_numeric(df_senti[col], errors='coerce')
df_senti = df_senti.dropna(subset=EMOTION_COLS)

df_daily_senti = df_senti.groupby('DATE').agg({**{col: ['mean', 'std', 'max', 'min'] for col in EMOTION_COLS}}).reset_index()
df_daily_senti.columns = ['DATE'] + [f"{col[0]}_{col[1]}" for col in df_daily_senti.columns[1:]]
df = pd.merge(df_price, df_daily_senti, on='DATE', how='inner')
df = df.sort_values('DATE').reset_index(drop=True)
df = df.fillna(method='ffill').fillna(method='bfill')

all_features = []
for col in df.columns:
    if col not in ['DATE', 'Adj Close'] and df[col].dtype in ['float64', 'int64']:
        all_features.append(col)

price_cols = ['Close']
volume_cols = ['OBV']
indicator_cols = [col for col in all_features if any(
    x in col for x in ['RSI', 'MACD', 'BB', 'ATR', 'ADX', 'CCI', 'STOCH', 'MFI', 'WILLR', 'ROC', 'TSI', 'EMA', 'SMA'])]
sentiment_cols = [col for col in all_features if any(x in col for x in EMOTION_COLS)]

# --- Scale như lúc train ---
scaler_price = RobustScaler()
scaler_volume = StandardScaler()
scaler_indicators = MinMaxScaler()
scaler_sentiment = StandardScaler()
df[price_cols] = scaler_price.fit_transform(df[price_cols])
df[volume_cols] = scaler_volume.fit_transform(df[volume_cols])
df[indicator_cols] = scaler_indicators.fit_transform(df[indicator_cols])
if sentiment_cols:
    df[sentiment_cols] = scaler_sentiment.fit_transform(df[sentiment_cols])

# ---- Remove highly correlated features (giống lúc train) ----
correlation_matrix = df[all_features].corr()
high_corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.95:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)
import pickle

with open('enhanced_transformer_model.pt', 'rb') as f:
    selected_features = pickle.load(f)
print(f"Test with selected_features ({len(selected_features)}):", selected_features)


# ==== Tạo sliding window ====
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

X, y = create_enhanced_sequences(df, selected_features, WINDOW, prediction_horizon=1)
X_test = torch.tensor(X, dtype=torch.float32)
y_test = torch.tensor(y, dtype=torch.float32)

# ==== ĐỊNH NGHĨA MODEL GIỐNG FILE sentin_informer.py ====
import torch.nn as nn

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
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
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

    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x)
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
        transformer_out = self.transformer(x)
        query = self.pool_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_out, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_out = pooled_out.squeeze(1)
        price_pred = self.output_layers(pooled_out).squeeze(-1)
        return price_pred

# ==== LOAD MODEL ====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedTimeSeriesTransformer(
    feature_dim=len(selected_features), window=WINDOW,
    d_model=128, nhead=16, num_layers=6, dropout=0.2
).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

# ==== DỰ ĐOÁN ====
with torch.no_grad():
    y_pred = model(X_test.to(DEVICE)).cpu().numpy()
    y_true = y_test.cpu().numpy()

# ==== INVERSE SCALER ĐỂ RA GIÁ GỐC ====
def inverse_close_transform(val):
    close_idx = price_cols.index('Close')
    arr = np.zeros((len(val), len(price_cols)))
    arr[:, close_idx] = val.reshape(-1)
    return scaler_price.inverse_transform(arr)[:, close_idx]
y_true_real = inverse_close_transform(y_true.reshape(-1, 1))
y_pred_real = inverse_close_transform(y_pred.reshape(-1, 1))

# ==== TÍNH DIRECTION & METRICS ====
def calc_direction_with_threshold(arr, threshold=0.001):
    changes = np.diff(arr)
    return (changes > threshold).astype(int)
y_true_cls = calc_direction_with_threshold(y_true_real, threshold=np.std(y_true_real) * 0.1)
y_pred_cls = calc_direction_with_threshold(y_pred_real, threshold=np.std(y_pred_real) * 0.1)
min_len = min(len(y_true_cls), len(y_pred_cls))
y_true_cls = y_true_cls[:min_len]
y_pred_cls = y_pred_cls[:min_len]

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

print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ==== VẼ BIỂU ĐỒ ====
plt.figure(figsize=(12, 5))
plt.plot(y_true_real, label='True Close')
plt.plot(y_pred_real, label='Predicted Close')
plt.legend()
plt.title('S&P 500 Close Price Prediction (Informer/Transformer)')
plt.show()