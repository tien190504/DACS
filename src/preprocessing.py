import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from config import EMOTION_COLS


def add_enhanced_indicators(df):
    """Add enhanced technical indicators to price data."""
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


def process_sentiment_data(df_senti):
    """Process sentiment data with enhanced features."""
    # Convert date and clean data
    df_senti['DATE'] = pd.to_datetime(df_senti['DATE'], errors='coerce')
    df_senti = df_senti.dropna(subset=['DATE'])

    # Process emotions with additional features
    for col in EMOTION_COLS:
        df_senti[col] = pd.to_numeric(df_senti[col], errors='coerce')
    df_senti = df_senti.dropna(subset=EMOTION_COLS)

    # Enhanced sentiment aggregation
    df_daily_senti = df_senti.groupby('DATE').agg({
        **{col: ['mean', 'std', 'max', 'min'] for col in EMOTION_COLS}
    }).reset_index()

    df_daily_senti.columns = ['DATE'] + [f"{col[0]}_{col[1]}" for col in df_daily_senti.columns[1:]]

    return df_daily_senti


def process_price_data(df_price):
    """Process price data with technical indicators."""
    df_price['DATE'] = pd.to_datetime(df_price['DATE'], errors='coerce')
    df_price = df_price.dropna(subset=['DATE'])
    df_price = add_enhanced_indicators(df_price)
    return df_price


def merge_and_clean_data(df_price, df_daily_senti):
    """Merge price and sentiment data and clean it."""
    # Merge data
    df = pd.merge(df_price, df_daily_senti, on='DATE', how='inner')
    df = df.sort_values('DATE').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(method='bfill')

    return df


def select_features(df):
    """Select relevant features for modeling."""
    all_features = []
    for col in df.columns:
        if col not in ['DATE', 'Adj Close'] and df[col].dtype in ['float64', 'int64']:
            all_features.append(col)

    print(f"Total features: {len(all_features)}")

    # Feature selection using correlation
    correlation_matrix = df[all_features].corr()
    high_corr_features = set()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.95:
                colname = correlation_matrix.columns[i]
                high_corr_features.add(colname)

    selected_features = [col for col in all_features if col not in high_corr_features]
    print(f"Selected features after correlation filtering: {len(selected_features)}")

    return selected_features


def scale_features(df, selected_features):
    """Scale features using different scalers for different feature types."""
    # Initialize scalers
    scaler_price = RobustScaler()
    scaler_volume = StandardScaler()
    scaler_indicators = MinMaxScaler()
    scaler_sentiment = StandardScaler()

    # Define feature groups
    price_cols = ['Open', 'High', 'Low', 'Close']
    volume_cols = ['Volume', 'OBV']
    indicator_cols = [col for col in selected_features if any(
        x in col for x in
        ['RSI', 'MACD', 'BB', 'ATR', 'ADX', 'CCI', 'STOCH', 'MFI', 'WILLR', 'ROC', 'TSI', 'EMA', 'SMA'])]
    sentiment_cols = [col for col in selected_features if any(x in col for x in EMOTION_COLS)]

    # Scale features
    if any(col in df.columns for col in price_cols):
        available_price_cols = [col for col in price_cols if col in df.columns]
        df[available_price_cols] = scaler_price.fit_transform(df[available_price_cols])

    if any(col in df.columns for col in volume_cols):
        available_volume_cols = [col for col in volume_cols if col in df.columns]
        df[available_volume_cols] = scaler_volume.fit_transform(df[available_volume_cols])

    if indicator_cols:
        df[indicator_cols] = scaler_indicators.fit_transform(df[indicator_cols])

    if sentiment_cols:
        df[sentiment_cols] = scaler_sentiment.fit_transform(df[sentiment_cols])

    scalers = {
        'price': scaler_price,
        'volume': scaler_volume,
        'indicators': scaler_indicators,
        'sentiment': scaler_sentiment,
        'price_cols': price_cols
    }

    return df, scalers


def preprocess_data(df_senti, df_price):
    """Complete preprocessing pipeline."""
    print("Starting preprocessing...")

    # Process individual datasets
    df_price = process_price_data(df_price)
    df_daily_senti = process_sentiment_data(df_senti)

    # Merge and clean data
    df = merge_and_clean_data(df_price, df_daily_senti)

    # Select features
    selected_features = select_features(df)

    # Scale features
    df, scalers = scale_features(df, selected_features)

    print("Preprocessing completed.")
    return df, selected_features, scalers