import torch
from torch import nn


class ConditioningAugmentation(nn.Module):
    """A layer applying augmentations to text embedding."""

    def __init__(self, text_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(text_dim, embedding_dim * 4, bias=True),
            nn.GLU(dim=1),
        )

    def encode(self, text_embedding: torch.Tensor):
        x = self.fc(text_embedding)
        mu, logvar = x[:, : self.embedding_dim], x[:, self.embedding_dim :]
        return mu, logvar

    def reparametrize(
        self, mu: torch.Tensor, logvar: torch.Tensor, device: torch.device = "cpu"
    ):
        std = 0.5 * torch.exp(logvar)
        eps = torch.normal(mean=0, std=1, size=std.size()).to(device)

        return mu + eps * std

    def forward(self, text_embedding: torch.Tensor):
        mu, logvar = self.encode(text_embedding)
        context = self.reparametrize(mu, logvar)

        return context, mu, logvar


def conv3x3(in_channels: int, out_channels: int):
    """A setup to have the shape of input and output."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
    )


def down_sampling(in_channels: int, out_channels: int):
    return nn.Sequential(
        conv3x3(in_channels, out_channels * 2),
        nn.BatchNorm2d(out_channels * 2),
        nn.GLU(dim=1),
    )
