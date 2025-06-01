"""
Test Script for S&P 500 Price Prediction Model
This script loads and tests the trained model with comprehensive evaluation
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import your existing modules
from config import MODEL_SAVE_PATH, DEVICE, WINDOW, PREDICTION_HORIZON
from data_loader import load_data, create_enhanced_sequences, split_data, create_data_loaders
from preprocessing import preprocess_data
from model import create_model
from evaluation import comprehensive_evaluation, inverse_close_transform, evaluate_classification_metrics
from training import evaluate_model

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelTester:
    """Comprehensive model testing class."""

    def __init__(self, model_path=None):
        """Initialize the model tester."""
        self.model_path = model_path or MODEL_SAVE_PATH
        self.model = None
        self.test_loader = None
        self.scalers = None
        self.selected_features = None
        self.results = {}

    def load_and_prepare_data(self):
        """Load and prepare test data."""
        print("📁 Loading and preparing test data...")

        # Load raw data
        df_senti, df_price = load_data()

        # Preprocess data
        df_processed, self.selected_features, self.scalers = preprocess_data(df_senti, df_price)

        # Create sequences
        X, y = create_enhanced_sequences(
            df_processed, self.selected_features, WINDOW, PREDICTION_HORIZON
        )

        # Split data (we only need test set for testing)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Create test data loader
        _, _, self.test_loader = create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

        print(f"✅ Test data prepared: {len(X_test)} samples")
        print(f"✅ Features: {len(self.selected_features)}")

        return X_test, y_test

    def load_model(self):
        """Load the trained model."""
        print(f"🤖 Loading model from: {self.model_path}")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Create model architecture
        self.model = create_model(
            feature_dim=len(self.selected_features),
            window=WINDOW,
            d_model=128,  # Default values - adjust if you know the exact parameters
            nhead=16,
            num_layers=6,
            dropout=0.2,
            device=DEVICE
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.model.eval()

        print("✅ Model loaded successfully")

    def run_basic_test(self):
        """Run basic model test."""
        print("\n🔍 Running basic model test...")

        # Get test loss and predictions
        test_loss, y_pred, y_true = evaluate_model(
            self.model, self.test_loader, DEVICE, return_predictions=True
        )

        # Store results
        self.results['test_loss'] = test_loss
        self.results['y_pred_scaled'] = y_pred
        self.results['y_true_scaled'] = y_true

        print(f"✅ Test Loss: {test_loss:.6f}")

        # Inverse transform to real prices
        y_true_real = inverse_close_transform(
            y_true.reshape(-1, 1),
            self.scalers['price'],
            self.scalers['price_cols']
        )
        y_pred_real = inverse_close_transform(
            y_pred.reshape(-1, 1),
            self.scalers['price'],
            self.scalers['price_cols']
        )

        self.results['y_true_real'] = y_true_real
        self.results['y_pred_real'] = y_pred_real

        return y_true_real, y_pred_real

    def evaluate_performance(self):
        """Evaluate model performance with detailed metrics."""
        print("\n📊 Evaluating model performance...")

        y_true_real = self.results['y_true_real']
        y_pred_real = self.results['y_pred_real']

        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true_real, y_pred_real)
        mae = mean_absolute_error(y_true_real, y_pred_real)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100
        r2 = r2_score(y_true_real, y_pred_real)

        # Classification metrics (direction prediction)
        acc, f1, auc_roc, pr_auc, y_true_cls, y_pred_cls = evaluate_classification_metrics(
            y_true_real, y_pred_real
        )

        # Store metrics
        self.results.update({
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': r2,
            'accuracy': acc,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'pr_auc': pr_auc,
            'y_true_cls': y_true_cls,
            'y_pred_cls': y_pred_cls
        })

        # Print results
        print(f"\n{'=' * 50}")
        print(f"📈 REGRESSION METRICS")
        print(f"{'=' * 50}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"R² Score: {r2:.4f}")

        print(f"\n{'=' * 50}")
        print(f"🎯 CLASSIFICATION METRICS (Direction)")
        print(f"{'=' * 50}")
        print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        return self.results

    def analyze_predictions(self):
        """Analyze prediction patterns and errors."""
        print("\n🔬 Analyzing prediction patterns...")

        y_true_real = self.results['y_true_real']
        y_pred_real = self.results['y_pred_real']

        # Calculate errors
        errors = y_pred_real - y_true_real
        percentage_errors = (errors / y_true_real) * 100

        # Error statistics
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'median_error': np.median(errors),
            'error_25_percentile': np.percentile(errors, 25),
            'error_75_percentile': np.percentile(errors, 75)
        }

        self.results['error_analysis'] = error_stats

        print(f"📊 Error Analysis:")
        print(f"   Mean Error: ${error_stats['mean_error']:.2f}")
        print(f"   Error Std: ${error_stats['std_error']:.2f}")
        print(f"   Max Error: ${error_stats['max_error']:.2f}")
        print(f"   Median Error: ${error_stats['median_error']:.2f}")

        return error_stats

    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\n📊 Creating visualizations...")

        y_true_real = self.results['y_true_real']
        y_pred_real = self.results['y_pred_real']
        errors = y_pred_real - y_true_real

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('S&P 500 Price Prediction Model - Test Results', fontsize=16, fontweight='bold')

        # 1. Price Prediction Comparison (Last 100 days)
        last_n = min(100, len(y_true_real))
        axes[0, 0].plot(y_true_real[-last_n:], label='Actual Price', linewidth=2, color='blue')
        axes[0, 0].plot(y_pred_real[-last_n:], label='Predicted Price', linewidth=2, color='red', alpha=0.8)
        axes[0, 0].set_title(f'Price Prediction (Last {last_n} days)', fontweight='bold')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Scatter Plot: Predicted vs Actual
        axes[0, 1].scatter(y_true_real, y_pred_real, alpha=0.6, s=20)
        min_price = min(y_true_real.min(), y_pred_real.min())
        max_price = max(y_true_real.max(), y_pred_real.max())
        axes[0, 1].plot([min_price, max_price], [min_price, max_price], 'r--', linewidth=2)
        axes[0, 1].set_title('Predicted vs Actual Prices', fontweight='bold')
        axes[0, 1].set_xlabel('Actual Price ($)')
        axes[0, 1].set_ylabel('Predicted Price ($)')
        axes[0, 1].grid(True, alpha=0.3)

        # Add R² to the plot
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true_real, y_pred_real)
        axes[0, 1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 3. Error Distribution
        axes[0, 2].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0, 2].axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2,
                           label=f'Mean Error: ${np.mean(errors):.2f}')
        axes[0, 2].set_title('Prediction Error Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Error ($)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Direction Accuracy Over Time
        if 'y_true_cls' in self.results and 'y_pred_cls' in self.results:
            y_true_cls = self.results['y_true_cls']
            y_pred_cls = self.results['y_pred_cls']

            # Calculate rolling accuracy
            window_size = 20
            rolling_acc = []
            for i in range(window_size, len(y_true_cls)):
                acc = np.mean(y_true_cls[i - window_size:i] == y_pred_cls[i - window_size:i])
                rolling_acc.append(acc)

            axes[1, 0].plot(rolling_acc, linewidth=2, color='purple')
            axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random Guess')
            axes[1, 0].axhline(y=self.results['accuracy'], color='red', linestyle='-',
                               label=f'Overall Accuracy: {self.results["accuracy"]:.3f}')
            axes[1, 0].set_title(f'Direction Accuracy (Rolling {window_size}-day window)', fontweight='bold')
            axes[1, 0].set_xlabel('Days')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, 1)

        # 5. Error over Time
        axes[1, 1].plot(errors, alpha=0.7, linewidth=1, color='orange')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.8)
        axes[1, 1].axhline(y=np.mean(errors), color='red', linestyle='--', alpha=0.8,
                           label=f'Mean Error: ${np.mean(errors):.2f}')
        axes[1, 1].fill_between(range(len(errors)),
                                np.mean(errors) - np.std(errors),
                                np.mean(errors) + np.std(errors),
                                alpha=0.2, color='red', label='±1 Std Dev')
        axes[1, 1].set_title('Prediction Error Over Time', fontweight='bold')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Error ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Performance Summary (Text)
        axes[1, 2].axis('off')
        summary_text = f"""
        MODEL PERFORMANCE SUMMARY

        📊 Regression Metrics:
        • MAE: ${self.results['mae']:.2f}
        • RMSE: ${self.results['rmse']:.2f}
        • MAPE: {self.results['mape']:.2f}%
        • R²: {self.results['r2_score']:.4f}

        🎯 Classification Metrics:
        • Accuracy: {self.results['accuracy']:.4f} ({self.results['accuracy'] * 100:.2f}%)
        • F1-Score: {self.results['f1_score']:.4f}
        • AUC-ROC: {self.results['auc_roc']:.4f}

        📈 Test Loss: {self.results['test_loss']:.6f}

        🔍 Data Info:
        • Test Samples: {len(y_true_real):,}
        • Features: {len(self.selected_features)}
        • Window Size: {WINDOW}
        """

        axes[1, 2].text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                        transform=axes[1, 2].transAxes, fontfamily='monospace')

        plt.tight_layout()
        plt.show()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"model_test_results_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📁 Plot saved as: {plot_filename}")

    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📋 Generating test report...")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""
        {'=' * 80}
        S&P 500 PRICE PREDICTION MODEL - TEST REPORT
        {'=' * 80}

        Test Date: {timestamp}
        Model Path: {self.model_path}
        Device: {DEVICE}

        {'=' * 80}
        DATA INFORMATION
        {'=' * 80}
        • Test Samples: {len(self.results['y_true_real']):,}
        • Features Used: {len(self.selected_features)}
        • Window Size: {WINDOW} days
        • Prediction Horizon: {PREDICTION_HORIZON} day(s)

        {'=' * 80}
        PERFORMANCE METRICS
        {'=' * 80}

        📊 REGRESSION METRICS:
        • Mean Absolute Error (MAE): ${self.results['mae']:.2f}
        • Root Mean Squared Error (RMSE): ${self.results['rmse']:.2f}
        • Mean Absolute Percentage Error (MAPE): {self.results['mape']:.2f}%
        • R² Score: {self.results['r2_score']:.4f}
        • Test Loss: {self.results['test_loss']:.6f}

        🎯 CLASSIFICATION METRICS (Direction Prediction):
        • Accuracy: {self.results['accuracy']:.4f} ({self.results['accuracy'] * 100:.2f}%)
        • F1-Score: {self.results['f1_score']:.4f}
        • AUC-ROC: {self.results['auc_roc']:.4f}
        • PR-AUC: {self.results['pr_auc']:.4f}

        📈 ERROR ANALYSIS:
        • Mean Error: ${self.results['error_analysis']['mean_error']:.2f}
        • Error Standard Deviation: ${self.results['error_analysis']['std_error']:.2f}
        • Maximum Absolute Error: ${self.results['error_analysis']['max_error']:.2f}
        • Median Error: ${self.results['error_analysis']['median_error']:.2f}

        {'=' * 80}
        PERFORMANCE ASSESSMENT
        {'=' * 80}
        """

        # Performance assessment
        if self.results['accuracy'] >= 0.8:
            report += "\n✅ EXCELLENT: Direction prediction accuracy ≥ 80%"
        elif self.results['accuracy'] >= 0.7:
            report += "\n✅ GOOD: Direction prediction accuracy ≥ 70%"
        elif self.results['accuracy'] >= 0.6:
            report += "\n⚠️  FAIR: Direction prediction accuracy ≥ 60%"
        else:
            report += "\n❌ POOR: Direction prediction accuracy < 60%"

        if self.results['mape'] <= 5:
            report += "\n✅ EXCELLENT: MAPE ≤ 5%"
        elif self.results['mape'] <= 10:
            report += "\n✅ GOOD: MAPE ≤ 10%"
        elif self.results['mape'] <= 15:
            report += "\n⚠️  FAIR: MAPE ≤ 15%"
        else:
            report += "\n❌ POOR: MAPE > 15%"

        if self.results['r2_score'] >= 0.8:
            report += "\n✅ EXCELLENT: R² ≥ 0.8"
        elif self.results['r2_score'] >= 0.6:
            report += "\n✅ GOOD: R² ≥ 0.6"
        elif self.results['r2_score'] >= 0.4:
            report += "\n⚠️  FAIR: R² ≥ 0.4"
        else:
            report += "\n❌ POOR: R² < 0.4"

        report += f"\n\n{'=' * 80}\n"

        print(report)

        # Save report to file
        report_filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(report)
        print(f"📁 Report saved as: {report_filename}")

        return report

    def run_complete_test(self):
        """Run complete model testing pipeline."""
        print("🚀 STARTING COMPLETE MODEL TEST")
        print("=" * 60)

        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()

            # Step 2: Load model
            self.load_model()

            # Step 3: Run basic test
            self.run_basic_test()

            # Step 4: Evaluate performance
            self.evaluate_performance()

            # Step 5: Analyze predictions
            self.analyze_predictions()

            # Step 6: Create visualizations
            self.create_visualizations()

            # Step 7: Generate report
            self.generate_test_report()

            print("\n🎉 MODEL TESTING COMPLETED SUCCESSFULLY!")
            print("=" * 60)

            return self.results

        except Exception as e:
            print(f"\n❌ Error during testing: {str(e)}")
            raise e


def main():
    """Main function to run model testing."""
    print("🔬 S&P 500 PRICE PREDICTION MODEL TESTER")
    print("=" * 60)

    # Check if model file exists
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ Model file not found: {MODEL_SAVE_PATH}")
        print("Please ensure the model file exists or update MODEL_SAVE_PATH in config.py")
        return

    # Initialize tester
    tester = ModelTester(MODEL_SAVE_PATH)

    # Run complete test
    results = tester.run_complete_test()

    return results


if __name__ == "__main__":
    # Run the test
    test_results = main()