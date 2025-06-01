import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from config import EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_DELTA, MODEL_SAVE_PATH, DEVICE


def multi_task_loss(price_pred, price_true, direction_logits=None, direction_true=None, alpha=0.7):
    """Multi-task loss combining regression and classification."""
    regression_loss_fn = nn.SmoothL1Loss()
    classification_loss_fn = nn.CrossEntropyLoss()

    reg_loss = regression_loss_fn(price_pred, price_true)

    if direction_logits is not None and direction_true is not None:
        cls_loss = classification_loss_fn(direction_logits, direction_true)
        return alpha * reg_loss + (1 - alpha) * cls_loss, reg_loss, cls_loss

    return reg_loss, reg_loss, torch.tensor(0.0)


def get_direction_labels(y_batch):
    """Generate direction labels for classification task."""
    directions = torch.zeros(len(y_batch), dtype=torch.long, device=y_batch.device)
    return directions


def evaluate_model(model, data_loader, device, return_predictions=False):
    """Evaluate model on validation/test data."""
    model.eval()
    regression_loss_fn = nn.SmoothL1Loss()
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


def train_model(model, train_loader, val_loader, device=DEVICE):
    """Train the model with enhanced training loop."""
    print("Starting enhanced training...")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)

            # Forward pass with multi-task learning
            price_pred, direction_logits = model(xb, return_direction=True)
            direction_labels = get_direction_labels(yb)

            # Calculate multi-task loss
            loss, reg_loss, cls_loss = multi_task_loss(
                price_pred, yb, direction_logits, direction_labels
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item() * xb.size(0)
            total_reg_loss += reg_loss.item() * xb.size(0)
            total_cls_loss += cls_loss.item() * xb.size(0)

        # Calculate average losses
        train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, val_loader, device)

        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step()

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  Reg Loss: {total_reg_loss / len(train_loader.dataset):.6f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
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

    # Load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    return model, train_losses, val_losses, best_val_loss