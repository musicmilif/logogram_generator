import math
import torch
from torch import nn

from model.general_layers import ConditioningAugmentation, conv3x3


def upsamping_block(in_channels: int, out_channels: int):
    """Double the size of width and height"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_channels, out_channels * 2),
        nn.BatchNorm2d(out_channels * 2),
        nn.GLU(dim=1),  # [batch, channel, width, height]
    )


class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.residual_block = nn.Sequential(
            conv3x3(num_channels, num_channels * 2),
            nn.BatchNorm2d(num_channels * 2),
            nn.GLU(dim=1),
            conv3x3(num_channels, num_channels),
            nn.BatchNorm2d(num_channels),
        )

    def forward(self, x):
        residual = x
        output = self.residual_block(x)

        return output + residual


class InitialGenerator(nn.Module):
    def __init__(
        self,
        gen_channels: int,
        noise_dim: int,
        embedding_dim: int,
        kernel_size: int = 4,
    ):
        """
        The initial generator, will scale up the shape to 16 times large and decrease
        the number of channels to 1/16 of input channel size.
        """
        super().__init__()
        self.gen_channels = gen_channels
        self.kernel_size = kernel_size
        in_channels = noise_dim + embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(
                in_channels,
                gen_channels * kernel_size * kernel_size * 2,
                bias=False,
            ),
            nn.BatchNorm1d(gen_channels * kernel_size * kernel_size * 2),
            nn.GLU(dim=1),
        )

        self.up_samplings = nn.Sequential(
            upsamping_block(gen_channels, gen_channels // 2),
            upsamping_block(gen_channels // 2, gen_channels // 4),
            upsamping_block(gen_channels // 4, gen_channels // 8),
            upsamping_block(gen_channels // 8, gen_channels // 16),
        )

    def forward(self, context, rand):
        x = torch.cat((context, rand), axis=1)
        x = self.fc(x).view((-1, self.gen_channels, self.kernel_size, self.kernel_size))
        x = self.up_samplings(x)

        return x


class RepeatedGenerator(nn.Module):
    def __init__(self, gen_channels: int, embedding_dim: int, num_residual: int = 2):
        """
        The repeated generator in the tree structure, will scale up the shape to 2
        times large, and decrease the number of channel to 1/2 of input channel size.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.joint_block = nn.Sequential(
            conv3x3(embedding_dim + gen_channels, gen_channels * 2),
            nn.BatchNorm2d(gen_channels * 2),
            nn.GLU(dim=1),
        )
        self.residual_blocks = nn.Sequential(
            *[ResBlock(gen_channels) for _ in range(num_residual)]
        )
        self.upsampling_block = upsamping_block(gen_channels, gen_channels // 2)

    def forward(self, img, context):
        img_size = img.size(2)
        context = context.view(-1, self.embedding_dim, 1, 1).repeat(
            1, 1, img_size, img_size
        )

        inputs = torch.cat((context, img), 1)
        inputs = self.joint_block(inputs)
        inputs = self.residual_blocks(inputs)
        outputs = self.upsampling_block(inputs)

        return outputs


class ImageGenerator(nn.Module):
    def __init__(self, gen_channels: int):
        super().__init__()
        self.gen_channels = nn.Sequential(
            conv3x3(gen_channels, 1),  # Gray scale in logogram case
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen_channels(x)


class GeneratorNet(nn.Module):
    def __init__(
        self,
        text_dim: int,
        embedding_dim: int,
        noise_dim: int,
        gen_channels: int,
        resolution: int,
    ):
        super().__init__()
        self.num_layers = int(math.log(resolution, 2)) - 5
        self.ca_net = ConditioningAugmentation(text_dim, embedding_dim)
        self.generator_layers = [
            InitialGenerator(gen_channels * 16, noise_dim, embedding_dim)
        ]
        self.image_layers = [ImageGenerator(gen_channels)]

        for i in range(1, self.num_layers):
            self.generator_layers.append(
                RepeatedGenerator(gen_channels // (2 ** (i - 1)), embedding_dim)
            )
            self.image_layers.append(ImageGenerator(gen_channels // (2 ** i)))

    def forward(self, text_embedding, noise):
        fake_images = []
        context, mu_, logvar_ = self.ca_net(text_embedding)

        output = self.generator_layers[0](context, noise)
        fake_images.append(self.image_layers[0](output))

        for i in range(1, self.num_layers):
            output = self.generator_layers[i](output, context)
            fake_images.append(self.image_layers[i](output))

        return fake_images, mu_, logvar_
