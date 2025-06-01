import os
import sys
import torch
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from flask_cors import CORS
import logging
from sklearn.preprocessing import StandardScaler
import pickle

warnings.filterwarnings('ignore')

# Import your model components (adjust paths as needed)
from src.model import EnhancedTimeSeriesTransformer
from src.config import (
    WINDOW, D_MODEL, N_HEAD, N_LAYERS, DROPOUT,
    MODEL_SAVE_PATH, DEVICE, PRICE_FEATURES, EMOTION_COLS,
    SENTI_FILE
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedPredictionService:
    """Enhanced service class to handle real data integration"""

    def __init__(self):
        self.model = None
        self.price_scaler = None
        self.feature_scalers = {}
        self.sentiment_data = None
        self.last_data = None
        self.last_update = None
        self.load_sentiment_data()
        self.load_model()

    def load_sentiment_data(self):
        """Load and preprocess sentiment data from CSV"""
        try:
            if not os.path.exists(SENTI_FILE):
                logger.warning(f"Sentiment file not found: {SENTI_FILE}")
                return False

            # Load sentiment data
            self.sentiment_data = pd.read_csv(SENTI_FILE)

            # Convert date column to datetime (adjust column name as needed)
            if 'date' in self.sentiment_data.columns:
                self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
            elif 'Date' in self.sentiment_data.columns:
                self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['Date'])

            # Set date as index for easier lookup
            self.sentiment_data.set_index('date', inplace=True)

            logger.info(f"✅ Sentiment data loaded: {len(self.sentiment_data)} records")
            logger.info(f"   Date range: {self.sentiment_data.index.min()} to {self.sentiment_data.index.max()}")
            logger.info(f"   Columns: {list(self.sentiment_data.columns)}")

            return True

        except Exception as e:
            logger.error(f"Error loading sentiment data: {e}")
            return False

    def load_model(self):
        """Load the trained model and preprocessing components"""
        try:
            if not os.path.exists(MODEL_SAVE_PATH):
                logger.error(f"Model file not found: {MODEL_SAVE_PATH}")
                return False

            # Create model with correct feature dimensions
            feature_dim = len(PRICE_FEATURES) + len(EMOTION_COLS)

            self.model = EnhancedTimeSeriesTransformer(
                feature_dim=feature_dim,
                window=WINDOW,
                d_model=D_MODEL,
                nhead=N_HEAD,
                num_layers=N_LAYERS,
                dropout=DROPOUT
            ).to(DEVICE)

            # Load trained weights
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
            self.model.eval()

            # Load scalers if they exist
            self.load_scalers()

            logger.info(f"✅ Model loaded successfully with {feature_dim} features")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_scalers(self):
        """Load preprocessing scalers"""
        try:
            scaler_path = MODEL_SAVE_PATH.replace('.pt', '_scalers.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.price_scaler = scalers.get('price_scaler')
                    self.feature_scalers = scalers.get('feature_scalers', {})
                logger.info("✅ Scalers loaded successfully")
            else:
                logger.warning("No scalers found - will use default scaling")
                self.price_scaler = StandardScaler()

        except Exception as e:
            logger.error(f"Error loading scalers: {e}")

    def get_latest_market_data(self, days=90):
        """Fetch latest S&P 500 data from yfinance"""
        try:
            # Fetch SPY data
            spy = yf.Ticker("SPY")
            hist = spy.history(period=f"{days}d")

            if hist.empty:
                logger.error("No data fetched from Yahoo Finance")
                return None

            # Reset index to get date as column
            hist.reset_index(inplace=True)

            logger.info(f"✅ Market data fetched: {len(hist)} days")
            logger.info(f"   Date range: {hist['Date'].min()} to {hist['Date'].max()}")

            return hist

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def merge_sentiment_with_prices(self, price_data):
        """Merge sentiment data with price data by date"""
        try:
            merged_data = []

            for _, row in price_data.iterrows():
                date = row['Date'].date()

                # Get price features
                price_features = {}
                for col in PRICE_FEATURES:
                    price_features[col] = row[col]

                # Get sentiment features for this date
                sentiment_features = self.get_sentiment_for_date(date)

                # Combine features
                combined_row = {**price_features, **sentiment_features, 'Date': date}
                merged_data.append(combined_row)

            merged_df = pd.DataFrame(merged_data)
            merged_df['Date'] = pd.to_datetime(merged_df['Date'])

            logger.info(f"✅ Data merged: {len(merged_df)} records")
            return merged_df

        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return None

    def get_sentiment_for_date(self, target_date):
        """Get sentiment features for a specific date"""
        try:
            # Convert to datetime if needed
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date).date()
            elif hasattr(target_date, 'date'):
                target_date = target_date.date()

            # Find sentiment data for this date (with fallback)
            sentiment_row = None

            # Try exact date match first
            for idx in self.sentiment_data.index:
                if idx.date() == target_date:
                    sentiment_row = self.sentiment_data.loc[idx]
                    break

            # If no exact match, find closest date within 7 days
            if sentiment_row is None:
                best_match = None
                min_diff = float('inf')

                for idx in self.sentiment_data.index:
                    diff = abs((idx.date() - target_date).days)
                    if diff <= 7 and diff < min_diff:
                        min_diff = diff
                        best_match = idx

                if best_match is not None:
                    sentiment_row = self.sentiment_data.loc[best_match]

            # Create sentiment features dictionary
            sentiment_features = {}
            for col in EMOTION_COLS:
                if sentiment_row is not None and col in sentiment_row:
                    sentiment_features[col] = sentiment_row[col]
                else:
                    # Use default values if no sentiment data available
                    if col == 'label':
                        sentiment_features[col] = 0  # neutral
                    else:
                        sentiment_features[col] = 0.5  # neutral emotion

            return sentiment_features

        except Exception as e:
            logger.error(f"Error getting sentiment for date {target_date}: {e}")
            # Return default values
            sentiment_features = {}
            for col in EMOTION_COLS:
                if col == 'label':
                    sentiment_features[col] = 0
                else:
                    sentiment_features[col] = 0.5
            return sentiment_features

    def preprocess_for_prediction(self, data):
        """Preprocess merged data for model prediction"""
        try:
            # Select and order features correctly
            processed = pd.DataFrame()

            # Add price features
            for col in PRICE_FEATURES:
                if col in data.columns:
                    processed[col] = data[col]
                else:
                    logger.error(f"Missing price feature: {col}")
                    return None

            # Add emotion features
            for col in EMOTION_COLS:
                if col in data.columns:
                    processed[col] = data[col]
                else:
                    logger.warning(f"Missing emotion feature: {col}, using default")
                    if col == 'label':
                        processed[col] = 0
                    else:
                        processed[col] = 0.5

            # Fill any remaining NaN values
            processed = processed.fillna(method='ffill').fillna(method='bfill')

            # Apply scaling if scalers are available
            if self.price_scaler is not None:
                for col in PRICE_FEATURES:
                    if col in processed.columns:
                        processed[col] = self.price_scaler.fit_transform(processed[[col]]).flatten()

            logger.info(f"✅ Data preprocessed: {processed.shape}")
            logger.info(f"   Features: {list(processed.columns)}")

            return processed.values  # Return as numpy array

        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None

    def predict_next_price(self):
        """Make prediction for next day's closing price"""
        try:
            if self.model is None:
                return None, "Model not loaded"

            # Get latest market data
            market_data = self.get_latest_market_data()
            if market_data is None:
                return None, "Failed to fetch market data"

            # Merge with sentiment data
            merged_data = self.merge_sentiment_with_prices(market_data)
            if merged_data is None:
                return None, "Failed to merge sentiment data"

            # Preprocess data
            processed_data = self.preprocess_for_prediction(merged_data)
            if processed_data is None:
                return None, "Failed to preprocess data"

            if len(processed_data) < WINDOW:
                return None, f"Not enough data points. Need {WINDOW}, got {len(processed_data)}"

            # Prepare input sequence (last WINDOW days)
            sequence = processed_data[-WINDOW:]  # Shape: (WINDOW, feature_dim)
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)

            # Make prediction
            with torch.no_grad():
                prediction = self.model(sequence_tensor)
                pred_value = prediction.cpu().numpy()[0]

            # Get current price for comparison
            current_price = merged_data['Close'].iloc[-1]

            # Calculate prediction confidence (simple heuristic)
            recent_volatility = merged_data['Close'].tail(10).std()
            confidence = max(0.6, min(0.95, 1.0 - (recent_volatility / current_price)))

            result = {
                'prediction': float(pred_value),
                'current_price': float(current_price),
                'timestamp': datetime.now().isoformat(),
                'confidence': float(confidence),
                'data_points_used': len(processed_data),
                'last_market_date': merged_data['Date'].iloc[-1].strftime('%Y-%m-%d')
            }

            self.last_data = merged_data
            self.last_update = datetime.now()

            return result, None

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)

    def get_historical_data(self, days=30):
        """Get historical price and sentiment data"""
        try:
            if self.last_data is None:
                # Fetch fresh data
                market_data = self.get_latest_market_data(days + 30)
                if market_data is None:
                    return None
                merged_data = self.merge_sentiment_with_prices(market_data)
            else:
                merged_data = self.last_data

            if merged_data is None:
                return None

            # Get last N days
            recent_data = merged_data.tail(days)

            historical = []
            for _, row in recent_data.iterrows():
                historical.append({
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'price': float(row['Close']),
                    'sentiment_label': int(row.get('label', 0)),
                    'sentiment_scores': {
                        'joy': float(row.get('joy', 0.5)),
                        'sadness': float(row.get('sad', 0.5)),
                        'anger': float(row.get('anger', 0.5)),
                        'fear': float(row.get('fear', 0.5)),
                        'love': float(row.get('love', 0.5)),
                        'surprise': float(row.get('surprise', 0.5))
                    }
                })

            return historical

        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None


# Initialize enhanced prediction service
prediction_service = EnhancedPredictionService()


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'active',
        'message': 'Enhanced S&P 500 Prediction API is running',
        'model_loaded': prediction_service.model is not None,
        'sentiment_data_loaded': prediction_service.sentiment_data is not None,
        'sentiment_records': len(
            prediction_service.sentiment_data) if prediction_service.sentiment_data is not None else 0,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['GET'])
def predict():
    """Get next day price prediction with real data"""
    try:
        result, error = prediction_service.predict_next_price()

        if error:
            return jsonify({'error': error}), 500

        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/realtime_predict', methods=['GET'])
def realtime_predict():
    """Real-time prediction endpoint for dashboard"""
    try:
        result, error = prediction_service.predict_next_price()

        if error:
            return jsonify({'error': error}), 500

        return jsonify({
            'time': datetime.now().strftime('%H:%M:%S'),
            'pred': result['prediction'],
            'actual_price': result['current_price'],
            'confidence': result['confidence'],
            'timestamp': result['timestamp'],
            'last_market_date': result.get('last_market_date')
        })

    except Exception as e:
        logger.error(f"Real-time prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical_data', methods=['GET'])
def historical_data():
    """Get historical price and sentiment data"""
    try:
        days = request.args.get('days', 30, type=int)
        data = prediction_service.get_historical_data(days)

        if data is None:
            return jsonify({'error': 'Failed to fetch historical data'}), 500

        return jsonify({
            'data': data,
            'count': len(data)
        })

    except Exception as e:
        logger.error(f"Historical data error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/sentiment_analysis', methods=['GET'])
def sentiment_analysis():
    """Get recent sentiment analysis summary"""
    try:
        if prediction_service.sentiment_data is None:
            return jsonify({'error': 'Sentiment data not available'}), 500

        # Get recent sentiment data (last 30 days)
        recent_data = prediction_service.sentiment_data.tail(30)

        # Calculate sentiment statistics
        sentiment_summary = {
            'total_records': len(recent_data),
            'avg_emotions': {
                'joy': float(recent_data['joy'].mean()) if 'joy' in recent_data else 0,
                'sadness': float(recent_data['sad'].mean()) if 'sad' in recent_data else 0,
                'anger': float(recent_data['anger'].mean()) if 'anger' in recent_data else 0,
                'fear': float(recent_data['fear'].mean()) if 'fear' in recent_data else 0,
                'love': float(recent_data['love'].mean()) if 'love' in recent_data else 0,
                'surprise': float(recent_data['surprise'].mean()) if 'surprise' in recent_data else 0
            },
            'sentiment_distribution': {
                'positive': int((recent_data['label'] == 1).sum()) if 'label' in recent_data else 0,
                'negative': int((recent_data['label'] == 0).sum()) if 'label' in recent_data else 0
            }
        }

        return jsonify(sentiment_summary)

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Starting Enhanced S&P 500 Prediction API Server")
    print("=" * 70)
    print(f"📊 Model loaded: {prediction_service.model is not None}")
    print(f"📰 Sentiment data loaded: {prediction_service.sentiment_data is not None}")
    if prediction_service.sentiment_data is not None:
        print(f"📈 Sentiment records: {len(prediction_service.sentiment_data)}")
    print(f"🔧 Device: {DEVICE}")
    print(f"📡 Server starting on http://localhost:5000")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)