import math
import torch
from torch import nn

from model.general_layers import conv3x3


def forward_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        conv3x3(in_channels, out_channels),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def down_sampling_image_block(dis_channels: int, reduction_ratio: int):
    module = [
        nn.Conv2d(1, dis_channels, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
    ]
    in_channels, out_channels = dis_channels, dis_channels * 2

    for _ in range(int(math.log(reduction_ratio, 2)) - 1):
        module += [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels, out_channels = out_channels, out_channels * 2

    in_channels, out_channels = out_channels // 2, in_channels // 2
    for _ in range(int(math.log(reduction_ratio, 2)) - 4):
        module += [
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels, out_channels = out_channels, out_channels // 2

    return nn.Sequential(*module)


class DiscriminatorResN(nn.Module):
    def __init__(self, dis_channels: int, embedding_dim: int, resolution: int):
        super().__init__()
        self.dis_channels = dis_channels
        self.embedding_dim = embedding_dim

        self.encode_image = down_sampling_image_block(dis_channels, resolution // 4)
        self.joint_block = forward_block(
            dis_channels * 8 + embedding_dim, dis_channels * 8
        )
        self.cond_logits = nn.Sequential(
            nn.Conv2d(dis_channels * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(dis_channels * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, x, context):
        x = self.encode_image(x)
        context = context.view(-1, self.embedding_dim, 1, 1).repeat(1, 1, 4, 4)

        x = torch.cat((context, x), 1)
        x = self.joint_block(x)

        return self.cond_logits(x).view(-1), self.uncond_logits(x).view(-1)
