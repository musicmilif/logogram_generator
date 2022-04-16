import os
from glob import glob
from typing import Any, Dict

import torch
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from utils import load_checkpoint, save_checkpoint


class KLDivLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        kl_ = input.pow(2).add_(target.exp()).mul_(-1).add_(1).add_(target)
        loss = torch.mean(kl_).mul_(-0.5)

        return loss


class GANTrainer:
    def __init__(
        self,
        models: Dict[str, Any],
        optimizers: Dict[str, Any],
        criterions: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        cond_weight: float,
    ):
        self.models = models
        self.optimizers = {
            "generator": optimizers["generator"](
                self.models["generator"].parameters(),
                lr=optimizer_params["gen_lr"],
                betas=(0.5, 0.999),
            ),
            "discriminator": [
                optimizers["discriminator"](
                    m.parameters(), lr=optimizer_params["dis_lr"], betas=(0.5, 0.999)
                )
                for m in self.models["discriminator"]
            ],
        }
        self.criterions = criterions
        self.cond_weight = cond_weight
        self.epoch = 0

    def train(self, data_loader: DataLoader, n_epochs: int, snapshot_at: int):
        for epoch in range(self.epoch + 1, self.epoch + n_epochs + 1):
            for words, pos_images, neg_images, noise in data_loader:
                self.models["generator"].eval()
                with torch.no_grad():
                    fake_images, mu_, logvar_ = self.models["generator"](words, noise)

                dis_loss = self.train_discriminator(
                    pos_images, neg_images, fake_images, mu_
                )
                gen_loss, kl_loss = self.train_generator(fake_images, mu_, logvar_)

            if epoch % snapshot_at == 0:
                save_checkpoint(
                    "check_points",
                    epoch=epoch,
                    models=self.models,
                    optimizer=self.optimizers,
                )

            print(
                f"Epoch: {epoch}/{n_epochs + self.epoch} | "
                f"Discriminator Loss: {dis_loss:.6f}\t"
                f"Generator Loss: {gen_loss:.6f}\tKL Divergence: {kl_loss:.6f}"
            )

    def train_discriminator(self, pos_images, neg_images, fake_images, mu_):
        total_loss = 0
        pos_label, neg_label = torch.ones(mu_.shape[0]), torch.zeros(mu_.shape[0])
        num_models = len(self.models["discriminator"])
        criterion = self.criterions["discriminator"]

        for i in range(num_models):
            self.models["discriminator"][i].train()
            pos_preds = self.models["discriminator"][i](pos_images[i], mu_.detach())
            neg_preds = self.models["discriminator"][i](neg_images[i], mu_.detach())
            fake_preds = self.models["discriminator"][i](
                fake_images[i].detach(), mu_.detach()
            )

            loss = (
                self.cond_weight * criterion(pos_preds[0], pos_label)
                + (1 - self.cond_weight) * criterion(pos_preds[1], pos_label)
                + self.cond_weight * criterion(neg_preds[0], neg_label)
                + (1 - self.cond_weight) * criterion(neg_preds[1], pos_label)
                + self.cond_weight * criterion(fake_preds[0], neg_label)
                + (1 - self.cond_weight) * criterion(fake_preds[1], neg_label)
            )
            loss.backward(retain_graph=True)
            self.optimizers["discriminator"][i].step()
            self.optimizers["discriminator"][i].zero_grad()

            total_loss += loss

        return total_loss

    def train_generator(self, fake_images, mu_, logvar_):
        total_loss = 0
        pos_label = torch.ones(mu_.shape[0])
        num_models = len(self.models["discriminator"])
        criterions = self.criterions["generator"]

        self.models["generator"].train()
        for i in range(num_models):
            self.models["discriminator"][i].train()
            fake_preds = self.models["discriminator"][i](fake_images[i], mu_)
            loss = (
                self.cond_weight * criterions(fake_preds[0], pos_label)
                + (1 - self.cond_weight) * criterions(fake_preds[1], pos_label)
            )
            total_loss += loss

        kl_divergance = KLDivLoss()(mu_, logvar_)
        total_loss += kl_divergance
        total_loss.backward()
        self.optimizers["generator"].step()
        self.optimizers["generator"].zero_grad()

        return total_loss, kl_divergance

    def load_checkpoint(self, path: str, reset_epoch: bool):
        # Load generator
        load_checkpoint(
            os.path.join(path, "generator.ckpt"),
            self.models["generator"],
            self.optimizers["generator"],
        )

        # Load discriminators
        for i in range(len(glob(os.path.join(path, "discriminator*")))):
            load_checkpoint(
                os.path.join(path, f"discriminator{i}.ckpt"),
                self.models["discriminator"][i],
                self.optimizers["discriminator"][i],
            )

        if not reset_epoch:
            self.epoch = int(path.split("/")[-1])
