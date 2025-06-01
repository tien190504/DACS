import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, \
    precision_recall_curve, auc, roc_curve
from training import evaluate_model


def inverse_close_transform(val, scaler_price, price_cols):
    """Inverse transform for Close price predictions."""
    close_idx = price_cols.index('Close')
    return scaler_price.inverse_transform(
        np.column_stack([
            np.zeros((len(val), close_idx)),
            val,
            np.zeros((len(val), len(price_cols) - close_idx - 1))
        ])
    )[:, close_idx]


def calc_direction_with_threshold(arr, threshold=0.001):
    """Calculate direction changes with threshold."""
    changes = np.diff(arr)
    return (changes > threshold).astype(int)


def calculate_trading_metrics(y_true, y_pred, transaction_cost=0.001):
    """Calculate trading simulation metrics."""
    true_returns = np.diff(y_true) / y_true[:-1]
    pred_returns = np.diff(y_pred) / y_pred[:-1]
    positions = np.where(pred_returns > 0, 1, -1)

    # Calculate transaction costs
    transaction_costs = transaction_cost * np.abs(np.diff(positions))
    transaction_costs = np.insert(transaction_costs, 0, 0)

    # Calculate strategy returns
    strategy_returns = positions * true_returns - transaction_costs

    # Calculate metrics
    total_return = np.prod(1 + strategy_returns) - 1
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(
        strategy_returns) > 0 else 0
    max_drawdown = np.min(np.cumsum(strategy_returns))

    return total_return, sharpe_ratio, max_drawdown


def evaluate_regression_metrics(y_true_real, y_pred_real):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true_real, y_pred_real)
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100

    return mse, mae, rmse, mape


def evaluate_classification_metrics(y_true_real, y_pred_real):
    """Calculate classification metrics for direction prediction."""
    # Calculate direction with adaptive threshold
    y_true_cls = calc_direction_with_threshold(y_true_real, threshold=np.std(y_true_real) * 0.1)
    y_pred_cls = calc_direction_with_threshold(y_pred_real, threshold=np.std(y_pred_real) * 0.1)

    # Ensure same length
    min_len = min(len(y_true_cls), len(y_pred_cls))
    y_true_cls = y_true_cls[:min_len]
    y_pred_cls = y_pred_cls[:min_len]

    # Calculate metrics
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

    return acc, f1, auc_roc, pr_auc, y_true_cls, y_pred_cls


def plot_results(y_true_real, y_pred_real, train_losses, val_losses, y_true_cls=None, y_pred_cls=None, auc_roc=None):
    """Plot comprehensive results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Price prediction plot
    axes[0, 0].plot(y_true_real[-200:], label='True Close', alpha=0.8, linewidth=2)
    axes[0, 0].plot(y_pred_real[-200:], label='Predicted Close', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('S&P 500 Close Price Prediction (Last 200 days)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Days')
    axes[0, 0].set_ylabel('Price ($)')

    # Residuals distribution
    residuals = y_true_real - y_pred_real
    axes[0, 1].hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 1].set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Zero Error')
    axes[0, 1].legend()

    # ROC Curve
    if y_true_cls is not None and y_pred_cls is not None and len(np.unique(y_true_cls)) > 1:
        fpr, tpr, _ = roc_curve(y_true_cls, y_pred_cls)
        axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.3f})', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve - Direction Prediction', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available',
                        ha='center', va='center', fontsize=12, transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('ROC Curve - Direction Prediction', fontsize=12, fontweight='bold')

    # Training history
    axes[1, 1].plot(train_losses, label='Training Loss', alpha=0.8, linewidth=2)
    axes[1, 1].plot(val_losses, label='Validation Loss', alpha=0.8, linewidth=2)
    axes[1, 1].set_title('Training History', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def comprehensive_evaluation(model, test_loader, scaler_price, price_cols, train_losses, val_losses, device):
    """Perform comprehensive model evaluation."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Get predictions
    test_loss, y_pred, y_true = evaluate_model(model, test_loader, device, return_predictions=True)

    # Inverse transform predictions
    y_true_real = inverse_close_transform(y_true.reshape(-1, 1), scaler_price, price_cols)
    y_pred_real = inverse_close_transform(y_pred.reshape(-1, 1), scaler_price, price_cols)

    # Regression metrics
    mse, mae, rmse, mape = evaluate_regression_metrics(y_true_real, y_pred_real)

    # Classification metrics
    acc, f1, auc_roc, pr_auc, y_true_cls, y_pred_cls = evaluate_classification_metrics(y_true_real, y_pred_real)

    # Trading metrics
    total_return, sharpe_ratio, max_drawdown = calculate_trading_metrics(y_true_real, y_pred_real)

    # Print results
    print(f"\n=== REGRESSION METRICS ===")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    print(f"\n=== CLASSIFICATION METRICS (Direction Prediction) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print(f"\n=== TRADING SIMULATION METRICS ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

    # Plot results
    plot_results(y_true_real, y_pred_real, train_losses, val_losses, y_true_cls, y_pred_cls, auc_roc)

    # Performance summary
    print(f"\n" + "=" * 60)
    print(f"PERFORMANCE SUMMARY")
    print(f"" + "=" * 60)
    print(f"Direction Accuracy: {acc:.4f} (Target: ≥0.8000)")
    print(f"Price RMSE: {rmse:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Achievement status
    if acc >= 0.8:
        print("\n🎉 TARGET ACCURACY ACHIEVED!")
        print("✅ Model performance meets the target criteria")
    else:
        print(f"\n⚠️  Target accuracy not achieved. Current: {acc:.4f}, Target: 0.8000")
        print("💡 Consider hyperparameter tuning or feature engineering")

    if sharpe_ratio > 1.0:
        print("✅ Excellent Sharpe ratio for trading strategy")
    elif sharpe_ratio > 0.5:
        print("✅ Good Sharpe ratio for trading strategy")
    else:
        print("⚠️  Low Sharpe ratio - trading strategy needs improvement")

    print("=" * 60)

    return {
        'test_loss': test_loss,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'accuracy': acc,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'y_true_real': y_true_real,
        'y_pred_real': y_pred_real
    }