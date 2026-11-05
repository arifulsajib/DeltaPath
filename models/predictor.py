
import torch
from torch import nn
from torch.nn import functional as F


class MlpPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, pred_type, spatial_border, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size) if pred_type == "classification" else nn.Identity(),
            nn.LeakyReLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Linear(hidden_size, output_size),
        )
        self.pred_type = pred_type
        self.spatial_border = nn.Parameter(torch.tensor(spatial_border), requires_grad=False)

    def forward(self, traj_h):
        return self.net(traj_h)

    def loss(self, traj_h, label, denormalize=False):
        pred = self.forward(traj_h)
        if self.pred_type == "regression":
            if denormalize:
                label = (label - self.spatial_border[0].unsqueeze(0)) / (
                    self.spatial_border[1] - self.spatial_border[0]
                ).unsqueeze(0)
            return F.mse_loss(pred, label)
        if self.pred_type == "classification":
            return F.cross_entropy(pred, label.long().squeeze(-1))
        raise NotImplementedError(f"No prediction type: {self.pred_type}.")
