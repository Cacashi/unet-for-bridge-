import torch
from torch import nn


class FiLMConditioner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, channels: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.projection = nn.Linear(hidden_dim, channels * 2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(self.backbone(tabular)).chunk(2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + gamma) + beta

