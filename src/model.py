import torch
import torch.nn as nn


class EnhancedTimeSeriesTransformer(nn.Module):
    """Enhanced Transformer model for time series prediction with multi-task learning."""

    def __init__(self, feature_dim, window, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.window = window
        self.feature_dim = feature_dim

        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(window, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, d_model))

        # Output layers for price prediction
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

        # Direction classifier for multi-task learning
        self.direction_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # up/down classification
        )

    def forward(self, x, return_direction=False):
        """Forward pass through the model."""
        batch_size = x.size(0)

        # Input projection and positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # Transformer encoding
        transformer_out = self.transformer(x)

        # Attention pooling
        query = self.pool_query.unsqueeze(0).expand(batch_size, -1, -1)
        pooled_out, _ = self.attention_pool(query, transformer_out, transformer_out)
        pooled_out = pooled_out.squeeze(1)

        # Price prediction
        price_pred = self.output_layers(pooled_out).squeeze(-1)

        if return_direction:
            # Direction classification for multi-task learning
            direction_logits = self.direction_classifier(pooled_out)
            return price_pred, direction_logits

        return price_pred


def create_model(feature_dim, window, d_model, nhead, num_layers, dropout, device):
    """Create and initialize the model."""
    model = EnhancedTimeSeriesTransformer(
        feature_dim=feature_dim,
        window=window,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model