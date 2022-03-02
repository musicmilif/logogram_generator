import math
from typing import Any, Dict

from model.generator import GeneratorNet
from model.discriminator import DiscriminatorResN


def stack_gan(resolution: int, configs: Dict[str, Any]):
    models = {
        "generator": GeneratorNet(
            configs["text_dim"],
            configs["embedding_dim"],
            configs["noise_dim"],
            configs["gen_channels"],
            resolution,
        ),
        "discriminator": [
            DiscriminatorResN(
                configs["dis_channels"], configs["embedding_dim"], 2 ** res
            ) for res in range(6, int(math.log(resolution, 2)) + 1)
        ]
    }
    return models
