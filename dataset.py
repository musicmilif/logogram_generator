import math
import os
import re
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = albu.Compose(
    [
        albu.ShiftScaleRotate(
            shift_limit=0.01,
            scale_limit=(-0.2, 0.1),
            rotate_limit=179,
            border_mode=cv2.BORDER_CONSTANT,
            value=255,
            p=0.95,
        ),
    ]
)


class ImageReader:
    def __init__(self, resolution: int):
        self.resolution = resolution

    def load_images(self, images_dir: str = "images"):
        df = []
        for path in glob(os.path.join(images_dir, "*.jpg")):
            row = {
                "words_embedding": self.get_words(Path(path).stem),
                "image_array": self.get_image(path),
            }
            df.append(row)

        return pd.DataFrame(df)

    def get_words(self, words: str):
        return re.findall("[A-Z][^A-Z_]*", words)

    def get_image(self, path: str):
        org_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        images = [
            cv2.resize(org_image, (2**res, 2**res))
            for res in range(6, int(math.log(self.resolution, 2)) + 1)
        ]

        return images


class LogogramDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        noise_dim: int,
        transformer: Optional[albu.Compose] = None,
        pretrain_model: str = "albert-base-v2",
    ):
        self.df = df
        self.noise_dim = noise_dim
        self.transformer = transformer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        self.langage_model = AutoModel.from_pretrained(pretrain_model)

    def __getitem__(self, idx: int):
        words, pos_images = self.df[["words_embedding", "image_array"]].iloc[idx]
        words = self._process_words(words)
        pos_images = self._process_images(pos_images)

        neg_idx = (np.random.randint(1, len(self.df)) + idx) % len(self.df)
        neg_images = self.df["image_array"].iloc[neg_idx]
        neg_images = self._process_images(neg_images)

        noise = torch.rand(self.noise_dim)

        return (words, pos_images, neg_images, noise)

    def __len__(self) -> int:
        return len(self.df)

    def _process_words(self, words: List[str], shift: bool = False) -> torch.Tensor:
        if shift:
            idx = np.random.randint(1, len(words))
            words = words[idx:] + words[:idx]

        words = self.tokenizer(words, padding=True, return_tensors="pt")
        words = self.langage_model(**words).last_hidden_state

        return torch.mean(words, axis=[0, 1])

    def _process_images(self, orig_images: List[np.ndarray]) -> List[torch.Tensor]:
        images = []
        for img in orig_images:
            img = self.transformer(image=img)["image"]
            img = torch.from_numpy(np.expand_dims(img, axis=0)) / 255
            images.append(img)

        return images
