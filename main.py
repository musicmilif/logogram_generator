import click
import yaml
from typing import Optional

import transformers
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import transformer, ImageReader, LogogramDataset
from model.gan import stack_gan
from trainer import GANTrainer

transformers.logging.set_verbosity_error()


@click.command()
@click.argument("checkpoint_path", type=click.Path(exists=True), required=False)
@click.option("--reset_epoch", is_flag=True)
def cli(checkpoint_path: Optional[str], reset_epoch: bool):
    with open("config.yaml") as f:
        configs = yaml.safe_load(f)

    log_df = ImageReader(resolution=configs["image"]["resolution"]).load_images(
        images_dir="./images"
    )

    log_dataset = LogogramDataset(
        df=log_df,
        noise_dim=configs["model"]["noise_dim"],
        transformer=transformer,
        pretrain_model="albert-base-v2",
    )
    train_loader = DataLoader(
        log_dataset, batch_size=configs["training"]["batch_size"], shuffle=True
    )

    model = GANTrainer(
        models=stack_gan(configs["image"]["resolution"], configs["model"]),
        optimizers={"generator": Adam, "discriminator": Adam},
        criterions={"generator": nn.BCELoss(), "discriminator": nn.BCELoss()},
        optimizer_params={
            "gen_lr": configs["training"]["gen_lr"],
            "dis_lr": configs["training"]["dis_lr"],
        },
        cond_weight=configs["training"]["cond_weight"],
    )
    if isinstance(checkpoint_path, str):
        model.load_checkpoint(checkpoint_path, reset_epoch)

    model.train(
        train_loader,
        configs["training"]["num_epochs"],
        configs["training"]["snapshot_at"],
    )

    return


if __name__ == "__main__":
    cli()
