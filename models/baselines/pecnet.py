import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_predictor import BasePredictor
from typing import Dict, Tuple, Optional


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: Tuple[int, ...] = (1024, 512),
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = (int(input_dim),) + tuple(int(x) for x in hidden_size) + (int(output_dim),)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.append(nn.ReLU())
                if float(dropout) > 0.0:
                    layers.append(nn.Dropout(p=float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PECNet(BasePredictor):
    """
    PECNet: Predicted Endpoint Conditioned Network
    参考论文: Mangalam et al. "PECNet: Trajectory Prediction with Planning-based 
    Endpoint Conditioned Network", CVPR 2020
    """
    def __init__(self, config: Dict):
        super(PECNet, self).__init__(config)
        self.fdim = int(config.get('fdim', 128))
        self.zdim = int(config.get('zdim', config.get('latent_dim', 16)))
        self.sigma = float(config.get('sigma', 1.0))
        self.dropout = float(config.get('dropout', 0.0))

        enc_past_size = tuple(config.get('enc_past_size', (1024, 512)))
        enc_dest_size = tuple(config.get('enc_dest_size', (256, 128)))
        enc_latent_size = tuple(config.get('enc_latent_size', (256, 128)))
        dec_size = tuple(config.get('dec_size', (256, 128)))
        predictor_size = tuple(config.get('predictor_size', (512, 256)))

        self.encoder_past = _MLP(
            input_dim=int(self.history_length) * 2,
            output_dim=int(self.fdim),
            hidden_size=enc_past_size,
            dropout=self.dropout,
        )
        self.encoder_dest = _MLP(
            input_dim=2,
            output_dim=int(self.fdim),
            hidden_size=enc_dest_size,
            dropout=self.dropout,
        )
        self.encoder_latent = _MLP(
            input_dim=int(self.fdim) * 2,
            output_dim=int(self.zdim) * 2,
            hidden_size=enc_latent_size,
            dropout=self.dropout,
        )
        self.decoder = _MLP(
            input_dim=int(self.fdim) + int(self.zdim),
            output_dim=2,
            hidden_size=dec_size,
            dropout=self.dropout,
        )
        self.predictor = _MLP(
            input_dim=int(self.fdim) * 2,
            output_dim=(int(self.future_length) - 1) * 2,
            hidden_size=predictor_size,
            dropout=self.dropout,
        )

    def forward(
        self,
        history: torch.Tensor,
        env_map: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        future: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = history.size(0)
        past = history.reshape(batch_size, -1)

        ftraj = self.encoder_past(past)

        if goal is not None:
            if goal.dim() == 3:
                dest = goal[:, 0, :]
            else:
                dest = goal
            latent = self.encoder_latent(torch.cat([ftraj, self.encoder_dest(dest)], dim=1))
            mu, log_var = torch.chunk(latent, 2, dim=1)
            z = self._reparameterize(mu, log_var)
        else:
            mu = torch.zeros(batch_size, self.zdim, device=history.device, dtype=history.dtype)
            log_var = torch.zeros(batch_size, self.zdim, device=history.device, dtype=history.dtype)
            z = torch.randn(batch_size, self.zdim, device=history.device, dtype=history.dtype) * self.sigma

        generated_dest = self.decoder(torch.cat([ftraj, z], dim=1))

        pred_mid_flat = self.predictor(torch.cat([ftraj, self.encoder_dest(generated_dest)], dim=1))
        pred_mid = pred_mid_flat.view(batch_size, int(self.future_length) - 1, 2)
        pred_traj = torch.cat([pred_mid, generated_dest.unsqueeze(1)], dim=1)

        return pred_traj, mu, log_var

    def predict(self, history: torch.Tensor, env_map: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
        batch_size = history.size(0)
        past = history.reshape(batch_size, -1)
        ftraj = self.encoder_past(past)

        all_samples = []
        for _ in range(int(num_samples)):
            z = torch.randn(batch_size, self.zdim, device=history.device, dtype=history.dtype) * self.sigma
            generated_dest = self.decoder(torch.cat([ftraj, z], dim=1))
            pred_mid_flat = self.predictor(torch.cat([ftraj, self.encoder_dest(generated_dest)], dim=1))
            pred_mid = pred_mid_flat.view(batch_size, int(self.future_length) - 1, 2)
            pred_traj = torch.cat([pred_mid, generated_dest.unsqueeze(1)], dim=1)
            all_samples.append(pred_traj.unsqueeze(1))

        return torch.cat(all_samples, dim=1)

    def predict_given_dest(self, history: torch.Tensor, env_map: torch.Tensor, dest: torch.Tensor) -> torch.Tensor:
        batch_size = history.size(0)
        past = history.reshape(batch_size, -1)
        ftraj = self.encoder_past(past)
        dest_feat = self.encoder_dest(dest)
        pred_mid_flat = self.predictor(torch.cat([ftraj, dest_feat], dim=1))
        pred_mid = pred_mid_flat.view(batch_size, int(self.future_length) - 1, 2)
        pred_traj = torch.cat([pred_mid, dest.unsqueeze(1)], dim=1)
        return pred_traj

    def _reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
