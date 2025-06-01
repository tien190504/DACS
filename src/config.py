import os
import torch

# --- Base paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')

# --- File paths & data ---
S_AND_P_500_NEWS_DATA = os.path.join(DATA_DIR, 'S&P500\\2')
S_AND_P_500_PRICES_DATA = os.path.join(DATA_DIR, 'S&P500\\1023')

SENTI_FILENAME = 'labeled_News_dataset_have_date.csv'
PRICE_FILENAME = 'sp500.csv'

SENTI_FILE = os.path.join(S_AND_P_500_NEWS_DATA, SENTI_FILENAME)
PRICE_FILE = os.path.join(S_AND_P_500_PRICES_DATA, PRICE_FILENAME)
# --- Enhanced Features ---
PRICE_FEATURES = [
    'Close'
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
ALPHA_MULTI_TASK_LOSS = 0.7 # Weight for regression loss in multi-task learning

# --- Enhanced Early Stopping ---
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_DELTA = 1e-6
MODEL_SAVE_PATH = 'optimized_model_20250601_080705.pt'