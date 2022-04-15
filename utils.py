import os
from collections import OrderedDict

import torch


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if "module" in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(dir_path, epoch, models, optimizer):
    base_dir = os.path.join(dir_path, str(epoch))
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Generator
    generator_dict = {
        "epoch": epoch,
        "state_dict": remove_redundant_keys(models["generator"].state_dict()),
        "optimizer": optimizer["generator"].state_dict(),
    }
    torch.save(generator_dict, os.path.join(base_dir, "generator.ckpt"))

    # Discriminator
    for i, (m, o) in enumerate(
        zip(models["discriminator"], optimizer["discriminator"])
    ):
        discriminator_dict = {
            "epoch": epoch,
            "state_dict": remove_redundant_keys(m.state_dict()),
            "optimizer": o.state_dict(),
        }
        torch.save(discriminator_dict, os.path.join(base_dir, f"discriminator{i}.ckpt"))


def load_checkpoint(path, model, optimizer=None):
    resume = torch.load(path)
    model.load_state_dict(remove_redundant_keys(resume["state_dict"]))
    if optimizer is not None:
        optimizer.load_state_dict(resume["optimizer"])
