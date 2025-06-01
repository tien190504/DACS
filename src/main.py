"""
Enhanced S&P 500 Price Prediction with Sentiment Analysis
Main execution script for the complete pipeline
"""

import sys
import os
import warnings
import torch
import numpy as np
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import project modules
from config import (
    WINDOW, PREDICTION_HORIZON, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    D_MODEL, N_HEAD, N_LAYERS, DROPOUT, DEVICE, MODEL_SAVE_PATH,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA
)
from data_loader import load_data, create_enhanced_sequences, split_data, create_data_loaders
from preprocessing import preprocess_data
from model import create_model
from training import train_model
from evaluation import comprehensive_evaluation


def print_banner():
    """Print project banner."""
    print("=" * 80)
    print("🚀 ENHANCED S&P 500 PRICE PREDICTION WITH SENTIMENT ANALYSIS")
    print("=" * 80)
    print("📊 Features: Multi-modal Transformer with Technical Indicators & Sentiment")
    print("🎯 Objective: Predict S&P 500 closing prices with >80% direction accuracy")
    print("🧠 Architecture: Enhanced Transformer with Multi-task Learning")
    print("=" * 80)
    print()


def print_system_info():
    """Print system and configuration information."""
    print("🔧 SYSTEM CONFIGURATION")
    print("-" * 40)
    print(f"Device: {DEVICE}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()


def print_model_config():
    """Print model configuration."""
    print("⚙️  MODEL CONFIGURATION")
    print("-" * 40)
    print(f"Window Size: {WINDOW}")
    print(f"Prediction Horizon: {PREDICTION_HORIZON}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Max Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Model Dimension: {D_MODEL}")
    print(f"Attention Heads: {N_HEAD}")
    print(f"Transformer Layers: {N_LAYERS}")
    print(f"Dropout Rate: {DROPOUT}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print()


def main():
    """Main execution function."""
    # Print information
    print_banner()
    print_system_info()
    print_model_config()

    try:
        # Phase 1: Data Loading
        print("📁 PHASE 1: DATA LOADING")
        print("-" * 40)
        df_senti, df_price = load_data()
        print(f"✅ Sentiment data loaded: {df_senti.shape}")
        print(f"✅ Price data loaded: {df_price.shape}")
        print()

        # Phase 2: Data Preprocessing
        print("🔄 PHASE 2: DATA PREPROCESSING")
        print("-" * 40)
        df_processed, selected_features, scalers = preprocess_data(df_senti, df_price)
        print(f"✅ Data processed: {df_processed.shape}")
        print(f"✅ Selected features: {len(selected_features)}")
        print(f"✅ Feature scaling completed")
        print()

        # Phase 3: Sequence Creation
        print("📈 PHASE 3: SEQUENCE CREATION")
        print("-" * 40)
        X, y = create_enhanced_sequences(df_processed, selected_features, WINDOW, PREDICTION_HORIZON)
        print(f"✅ Sequences created: X={X.shape}, y={y.shape}")

        # Data splitting
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, train_ratio=0.7, val_ratio=0.15)
        print(f"✅ Train set: {X_train.shape}, {y_train.shape}")
        print(f"✅ Validation set: {X_val.shape}, {y_val.shape}")
        print(f"✅ Test set: {X_test.shape}, {y_test.shape}")

        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)
        print(f"✅ Data loaders created")
        print()

        # Phase 4: Model Creation
        print("🧠 PHASE 4: MODEL CREATION")
        print("-" * 40)
        model = create_model(
            feature_dim=len(selected_features),
            window=WINDOW,
            d_model=D_MODEL,
            nhead=N_HEAD,
            num_layers=N_LAYERS,
            dropout=DROPOUT,
            device=DEVICE
        )
        print(f"✅ Model created and moved to {DEVICE}")
        print()

        # Phase 5: Model Training
        print("🎯 PHASE 5: MODEL TRAINING")
        print("-" * 40)
        start_time = datetime.now()

        model, train_losses, val_losses, best_val_loss = train_model(
            model, train_loader, val_loader, DEVICE
        )

        end_time = datetime.now()
        training_duration = end_time - start_time
        print(f"✅ Training completed in {training_duration}")
        print(f"✅ Best validation loss: {best_val_loss:.6f}")
        print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
        print()

        # Phase 6: Model Evaluation
        print("📊 PHASE 6: MODEL EVALUATION")
        print("-" * 40)
        results = comprehensive_evaluation(
            model=model,
            test_loader=test_loader,
            scaler_price=scalers['price'],
            price_cols=scalers['price_cols'],
            train_losses=train_losses,
            val_losses=val_losses,
            device=DEVICE
        )

        # Phase 7: Final Summary
        print("\n🎉 PHASE 7: EXECUTION SUMMARY")
        print("-" * 40)
        print(f"✅ Total execution time: {datetime.now() - start_time if 'start_time' in locals() else 'N/A'}")
        print(f"✅ Model performance:")
        print(f"   • Direction Accuracy: {results['accuracy']:.4f}")
        print(f"   • Price RMSE: {results['rmse']:.4f}")
        print(f"   • MAPE: {results['mape']:.2f}%")
        print(f"   • Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"   • Total Return: {results['total_return']:.2%}")

        # Success criteria check
        success_criteria = {
            'accuracy_target': results['accuracy'] >= 0.8,
            'rmse_reasonable': results['rmse'] < 100,  # Adjust based on your price scale
            'sharpe_positive': results['sharpe_ratio'] > 0
        }

        print(f"\n🎯 SUCCESS CRITERIA:")
        for criterion, passed in success_criteria.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   • {criterion.replace('_', ' ').title()}: {status}")

        overall_success = all(success_criteria.values())
        if overall_success:
            print(f"\n🎊 PROJECT COMPLETED SUCCESSFULLY!")
            print(f"🚀 Model is ready for deployment and trading simulation!")
        else:
            print(f"\n⚠️  Project completed with some targets not met.")
            print(f"💡 Consider hyperparameter tuning or additional feature engineering.")

        print("=" * 80)

        return model, results

    except Exception as e:
        print(f"\n❌ ERROR OCCURRED: {str(e)}")
        print(f"📋 Error Type: {type(e).__name__}")
        print(f"📍 Check your data files and configuration settings.")
        print("=" * 80)
        raise e


def run_inference_example(model, test_loader, scalers, device):
    """Run a simple inference example."""
    print("\n🔮 INFERENCE EXAMPLE")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Make predictions
            predictions = model(X_batch)

            # Convert back to original scale
            from evaluation import inverse_close_transform
            pred_real = inverse_close_transform(
                predictions[:5].cpu().numpy().reshape(-1, 1),
                scalers['price'],
                scalers['price_cols']
            )
            true_real = inverse_close_transform(
                y_batch[:5].cpu().numpy().reshape(-1, 1),
                scalers['price'],
                scalers['price_cols']
            )

            print("Sample Predictions vs Actual:")
            for i in range(5):
                print(f"  Predicted: ${pred_real[i]:.2f} | Actual: ${true_real[i]:.2f} | "
                      f"Error: ${abs(pred_real[i] - true_real[i]):.2f}")
            break

    print("✅ Inference example completed")


if __name__ == "__main__":
    print(f"🚀 Starting Enhanced S&P 500 Prediction Pipeline...")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Run main pipeline
        model, results = main()

        # Optional: Run inference example
        print("\n" + "=" * 60)
        response = input("Would you like to see an inference example? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            # Reload data for inference example
            from data_loader import load_data, create_enhanced_sequences, split_data, create_data_loaders
            from preprocessing import preprocess_data

            df_senti, df_price = load_data()
            df_processed, selected_features, scalers = preprocess_data(df_senti, df_price)
            X, y = create_enhanced_sequences(df_processed, selected_features, WINDOW, PREDICTION_HORIZON)
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
            train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test)

            run_inference_example(model, test_loader, scalers, DEVICE)

        print(f"\n🎉 All operations completed successfully!")
        print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print(f"\n⚠️  Execution interrupted by user.")
        print(f"💾 Partial results may have been saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error occurred: {str(e)}")
        sys.exit(1)