"""
GridSearchCV Fine-tuning for S&P 500 Price Prediction Model
This script performs hyperparameter optimization to improve model accuracy from 0.606
"""

import itertools
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# Import your existing modules
from config import DEVICE, MODEL_SAVE_PATH
from data_loader import load_data, create_enhanced_sequences, split_data, create_data_loaders
from preprocessing import preprocess_data
from model import create_model
from training import train_model, evaluate_model
from evaluation import evaluate_classification_metrics, inverse_close_transform


class GridSearchTransformer:
    """Grid Search implementation for Transformer hyperparameter tuning."""

    def __init__(self,
                 param_grid: Dict[str, List],
                 scoring_metric: str = 'accuracy',
                 cv_folds: int = 3,
                 early_stopping_patience: int = 15,
                 max_epochs: int = 100,
                 device: str = 'cuda'):
        """
        Initialize GridSearch for Transformer model.

        Args:
            param_grid: Dictionary of hyperparameters to search
            scoring_metric: Metric to optimize ('accuracy', 'f1', 'loss')
            cv_folds: Number of cross-validation folds
            early_stopping_patience: Early stopping patience
            max_epochs: Maximum training epochs
            device: Device to use ('cuda' or 'cpu')
        """
        self.param_grid = param_grid
        self.scoring_metric = scoring_metric
        self.cv_folds = cv_folds
        self.early_stopping_patience = early_stopping_patience
        self.max_epochs = max_epochs
        self.device = device

        # Results storage
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf if scoring_metric != 'loss' else np.inf
        self.best_model_ = None

    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """Create time-series aware cross-validation splits."""
        n_samples = len(X)
        fold_size = n_samples // self.cv_folds

        splits = []
        for i in range(self.cv_folds):
            # For time series, use expanding window approach
            train_end = fold_size * (i + 2)  # Expanding training set
            val_start = fold_size * (i + 1)
            val_end = fold_size * (i + 2)

            if train_end > n_samples:
                train_end = n_samples
            if val_end > n_samples:
                val_end = n_samples

            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)

            splits.append((train_idx, val_idx))

        return splits

    def _evaluate_params(self, params: Dict, X: np.ndarray, y: np.ndarray,
                         feature_dim: int, window: int, scalers: Dict) -> Dict:
        """Evaluate a single parameter combination."""
        print(f"\n🔍 Testing parameters: {params}")

        # Create CV splits
        cv_splits = self._create_cv_splits(X, y)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"  📁 Fold {fold + 1}/{self.cv_folds}")

            # Split data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            # Create data loaders
            train_loader_fold = DataLoader(
                TensorDataset(
                    torch.tensor(X_train_fold, dtype=torch.float32),
                    torch.tensor(y_train_fold, dtype=torch.float32)
                ),
                batch_size=params.get('batch_size', 64),
                shuffle=True
            )

            val_loader_fold = DataLoader(
                TensorDataset(
                    torch.tensor(X_val_fold, dtype=torch.float32),
                    torch.tensor(y_val_fold, dtype=torch.float32)
                ),
                batch_size=params.get('batch_size', 64),
                shuffle=False
            )

            try:
                # Create model with current parameters
                model = create_model(
                    feature_dim=feature_dim,
                    window=window,
                    d_model=params.get('d_model', 128),
                    nhead=params.get('nhead', 8),
                    num_layers=params.get('num_layers', 4),
                    dropout=params.get('dropout', 0.2),
                    device=self.device
                )

                # Train model with early stopping
                model, train_losses, val_losses, best_val_loss = self._train_fold(
                    model, train_loader_fold, val_loader_fold, params
                )

                # Evaluate model
                score = self._score_model(model, val_loader_fold, y_val_fold, scalers)
                fold_scores.append(score)

                print(f"    ✅ Fold {fold + 1} {self.scoring_metric}: {score:.4f}")

            except Exception as e:
                print(f"    ❌ Fold {fold + 1} failed: {str(e)}")
                fold_scores.append(-np.inf if self.scoring_metric != 'loss' else np.inf)

        # Calculate mean score across folds
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        result = {
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores
        }

        print(f"  📊 Mean {self.scoring_metric}: {mean_score:.4f} ± {std_score:.4f}")

        return result

    def _train_fold(self, model, train_loader, val_loader, params):
        """Train model for a single fold with early stopping."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params.get('learning_rate', 1e-4),
            weight_decay=params.get('weight_decay', 1e-5)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=1e-6
        )

        criterion = nn.SmoothL1Loss()

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(self.max_epochs):
            # Training phase
            model.train()
            total_loss = 0

            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            # Validation phase
            val_loss = evaluate_model(model, val_loader, self.device)

            train_losses.append(total_loss / len(train_loader))
            val_losses.append(val_loss)

            scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break

        # Load best model state
        model.load_state_dict(best_model_state)

        return model, train_losses, val_losses, best_val_loss

    def _score_model(self, model, val_loader, y_true, scalers):
        """Score model based on selected metric."""
        # Get predictions
        _, y_pred, y_true_tensor = evaluate_model(
            model, val_loader, self.device, return_predictions=True
        )

        # Convert to real scale
        y_true_real = inverse_close_transform(
            y_true.reshape(-1, 1), scalers['price'], scalers['price_cols']
        )
        y_pred_real = inverse_close_transform(
            y_pred.reshape(-1, 1), scalers['price'], scalers['price_cols']
        )

        if self.scoring_metric == 'accuracy':
            acc, _, _, _, _, _ = evaluate_classification_metrics(y_true_real, y_pred_real)
            return acc
        elif self.scoring_metric == 'f1':
            _, f1, _, _, _, _ = evaluate_classification_metrics(y_true_real, y_pred_real)
            return f1
        elif self.scoring_metric == 'loss':
            return np.mean((y_true_real - y_pred_real) ** 2)
        else:
            raise ValueError(f"Unknown scoring metric: {self.scoring_metric}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_dim: int,
            window: int, scalers: Dict):
        """Perform grid search to find best hyperparameters."""
        print("🚀 Starting GridSearch hyperparameter optimization...")
        print(f"📊 Parameter combinations to test: {len(list(ParameterGrid(self.param_grid)))}")
        print(f"🎯 Optimization metric: {self.scoring_metric}")
        print(f"📁 Cross-validation folds: {self.cv_folds}")

        start_time = datetime.now()

        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(self.param_grid))

        for i, params in enumerate(param_combinations):
            print(f"\n{'=' * 60}")
            print(f"🔄 Progress: {i + 1}/{len(param_combinations)}")
            print(f"⏱️  Elapsed time: {datetime.now() - start_time}")

            # Evaluate current parameter combination
            result = self._evaluate_params(params, X, y, feature_dim, window, scalers)
            self.results_.append(result)

            # Update best parameters
            current_score = result['mean_score']
            is_better = (
                current_score > self.best_score_ if self.scoring_metric != 'loss'
                else current_score < self.best_score_
            )

            if is_better:
                self.best_score_ = current_score
                self.best_params_ = params.copy()
                print(f"  🎯 New best {self.scoring_metric}: {current_score:.4f}")

        print(f"\n{'=' * 60}")
        print("✅ GridSearch completed!")
        print(f"⏱️  Total time: {datetime.now() - start_time}")
        print(f"🏆 Best {self.scoring_metric}: {self.best_score_:.4f}")
        print(f"🎯 Best parameters: {self.best_params_}")

        return self

    def save_results(self, filepath: str):
        """Save grid search results to file."""
        results_data = {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'all_results': self.results_,
            'scoring_metric': self.scoring_metric,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"📁 Results saved to: {filepath}")


def define_search_space():
    """Define hyperparameter search space."""
    param_grid = {
        # Model architecture parameters
        'd_model': [64, 128, 256],
        'nhead': [4, 8, 16],
        'num_layers': [2, 4, 6],
        'dropout': [0.1, 0.2, 0.3],

        # Training parameters
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
        'batch_size': [32, 64, 128],
        'weight_decay': [1e-6, 1e-5, 1e-4],
    }

    return param_grid


def run_comprehensive_grid_search():
    """Run comprehensive grid search for model optimization."""
    print("🎯 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Load and preprocess data
    print("📁 Loading and preprocessing data...")
    df_senti, df_price = load_data()
    df_processed, selected_features, scalers = preprocess_data(df_senti, df_price)

    # Create sequences
    WINDOW = 30
    PREDICTION_HORIZON = 1
    X, y = create_enhanced_sequences(df_processed, selected_features, WINDOW, PREDICTION_HORIZON)

    print(f"✅ Data prepared: X={X.shape}, y={y.shape}")
    print(f"✅ Features: {len(selected_features)}")

    # Define search space
    param_grid = define_search_space()

    # Initialize grid search
    grid_search = GridSearchTransformer(
        param_grid=param_grid,
        scoring_metric='accuracy',  # Change to 'f1' or 'loss' if needed
        cv_folds=3,
        early_stopping_patience=10,
        max_epochs=50,  # Reduced for faster search
        device=DEVICE
    )

    # Perform grid search
    grid_search.fit(X, y, len(selected_features), WINDOW, scalers)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"gridsearch_results_{timestamp}.json"
    grid_search.save_results(results_file)

    return grid_search


def run_focused_grid_search():
    """Run focused grid search with smaller parameter space for faster results."""
    print("🎯 FOCUSED HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)

    # Load and preprocess data
    print("📁 Loading and preprocessing data...")
    df_senti, df_price = load_data()
    df_processed, selected_features, scalers = preprocess_data(df_senti, df_price)

    # Create sequences
    WINDOW = 30
    PREDICTION_HORIZON = 1
    X, y = create_enhanced_sequences(df_processed, selected_features, WINDOW, PREDICTION_HORIZON)

    print(f"✅ Data prepared: X={X.shape}, y={y.shape}")

    # Focused parameter grid (faster execution)
    param_grid = {
        'd_model': [128, 256],
        'nhead': [8, 16],
        'num_layers': [4, 6],
        'dropout': [0.1, 0.2],
        'learning_rate': [5e-5, 1e-4],
        'batch_size': [64, 128],
        'weight_decay': [1e-5],
    }

    # Initialize grid search
    grid_search = GridSearchTransformer(
        param_grid=param_grid,
        scoring_metric='accuracy',
        cv_folds=3,
        early_stopping_patience=8,
        max_epochs=30,
        device=DEVICE
    )

    # Perform grid search
    grid_search.fit(X, y, len(selected_features), WINDOW, scalers)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"focused_gridsearch_results_{timestamp}.json"
    grid_search.save_results(results_file)

    return grid_search


def train_best_model(best_params: Dict):
    """Train final model with best parameters found."""
    print("🏆 TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("=" * 60)
    print(f"Best parameters: {best_params}")

    # Load and preprocess data
    df_senti, df_price = load_data()
    df_processed, selected_features, scalers = preprocess_data(df_senti, df_price)

    # Create sequences and split data
    WINDOW = 30
    X, y = create_enhanced_sequences(df_processed, selected_features, WINDOW, 1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Create data loaders with best batch size
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    # Override batch size in loaders
    batch_size = best_params.get('batch_size', 64)
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # Create model with best parameters
    model = create_model(
        feature_dim=len(selected_features),
        window=WINDOW,
        d_model=best_params.get('d_model', 128),
        nhead=best_params.get('nhead', 8),
        num_layers=best_params.get('num_layers', 4),
        dropout=best_params.get('dropout', 0.2),
        device=DEVICE
    )

    # Train with best parameters
    model, train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, DEVICE
    )

    # Evaluate final model
    from evaluation import comprehensive_evaluation
    results = comprehensive_evaluation(
        model, test_loader, scalers['price'], scalers['price_cols'],
        train_losses, val_losses, DEVICE
    )

    # Save optimized model
    optimized_model_path = f"optimized_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model.state_dict(), optimized_model_path)
    print(f"🎯 Optimized model saved to: {optimized_model_path}")

    return model, results


if __name__ == "__main__":
    print("🚀 HYPERPARAMETER OPTIMIZATION PIPELINE")
    print("=" * 80)

    try:
        # Choose search type
        print("Select optimization strategy:")
        print("1. Focused Grid Search (faster, ~30 minutes)")
        print("2. Comprehensive Grid Search (thorough, ~2-3 hours)")
        print("3. Load previous results and train best model")

        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            grid_search = run_focused_grid_search()

            # Train final model with best parameters
            print("\n" + "=" * 60)
            final_model, final_results = train_best_model(grid_search.best_params_)

        elif choice == "2":
            grid_search = run_comprehensive_grid_search()

            # Train final model with best parameters
            print("\n" + "=" * 60)
            final_model, final_results = train_best_model(grid_search.best_params_)

        elif choice == "3":
            results_file = input("Enter results file path: ").strip()
            with open(results_file, 'r') as f:
                saved_results = json.load(f)

            best_params = saved_results['best_params']
            final_model, final_results = train_best_model(best_params)

        else:
            print("Invalid choice!")
            exit(1)

        print("\n🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print(f"🎯 Final Accuracy: {final_results['accuracy']:.4f}")
        print(f"📈 Improvement from baseline (0.606): {final_results['accuracy'] - 0.606:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise e