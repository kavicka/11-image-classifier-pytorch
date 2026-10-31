"""CIFAR-10 and a 10-class Food-101 subset: datasets, splits, augmentations, loaders.

Both have a fixed validation split carved out of the training set. Choosing hyperparameters on the
test set would make the final number optimistic, so test is only touched by the final models.
"""

import tarfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torchvision.transforms import v2

DATA = Path("data")
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
FOOD_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
# the first 10 classes alphabetically: not hand-picked, so nobody can say the easy dishes were chosen
FOOD_CLASSES = (
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
)


def _finish(ops: list, mean, std) -> v2.Compose:
    return v2.Compose([v2.ToImage(), *ops, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


def cifar_transform(aug: str) -> v2.Compose:
    ops = {
        "none": [],
        "flip": [v2.RandomHorizontalFlip()],
        "crop_flip": [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()],
        "crop_flip_erase": [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()],
    }[aug]
    tf = _finish(ops, CIFAR_MEAN, CIFAR_STD)
    if aug == "crop_flip_erase":  # erasing runs last: it hides a patch of the normalised image
        tf = v2.Compose([tf, v2.RandomErasing(p=0.5, scale=(0.02, 0.2))])
    return tf


def _loader(ds: Dataset, batch: int, train: bool, workers: int) -> DataLoader:
    return DataLoader(ds, batch, shuffle=train, drop_last=train, num_workers=workers, persistent_workers=workers > 0)


def cifar_loaders(aug: str, batch: int = 128, workers: int = 6) -> dict[str, DataLoader]:
    root = str(DATA)
    train_aug = datasets.CIFAR10(root, train=True, download=True, transform=cifar_transform(aug))
    train_plain = datasets.CIFAR10(root, train=True, transform=cifar_transform("none"))
    test = datasets.CIFAR10(root, train=False, transform=cifar_transform("none"))
    idx = torch.randperm(50_000, generator=torch.Generator().manual_seed(0)).tolist()  # same split in every run
    return {
        "train": _loader(Subset(train_aug, idx[5_000:]), batch, True, workers),
        "val": _loader(Subset(train_plain, idx[:5_000]), 512, False, 2),
        "test": _loader(test, 512, False, 2),
    }


def ensure_food10() -> Path:
    """Download the 5 GB archive once, unpack only the 10 classes plus metadata, delete the archive."""
    root = DATA / "food-101"
    if (root / "meta" / "test.txt").exists():
        return root
    archive = DATA / "food-101.tar.gz"
    if not archive.exists():
        download_url(FOOD_URL, str(DATA), archive.name)
    prefixes = tuple(f"food-101/images/{c}/" for c in FOOD_CLASSES) + ("food-101/meta/",)
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name.startswith(prefixes):
                tar.extract(member, DATA, filter="data")
    archive.unlink()
    return root


class Food10(Dataset):
    def __init__(self, split: str, transform):
        root = ensure_food10()
        names = (root / "meta" / f"{split}.txt").read_text().split()
        self.items = [
            (root / "images" / f"{n}.jpg", FOOD_CLASSES.index(n.split("/")[0]))
            for n in names
            if n.split("/")[0] in FOOD_CLASSES
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, y = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), y


def food_transform(aug: str) -> v2.Compose:
    if aug == "none":  # eval transform, also used to train the "no augmentation" runs
        return _finish([v2.Resize(256, antialias=True), v2.CenterCrop(224)], IMAGENET_MEAN, IMAGENET_STD)
    ops = [v2.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True), v2.RandomHorizontalFlip()]
    if aug == "strong":
        ops += [v2.ColorJitter(0.3, 0.3, 0.3, 0.05), v2.RandomRotation(15)]
    return _finish(ops, IMAGENET_MEAN, IMAGENET_STD)


def food_loaders(aug: str, batch: int = 64, workers: int = 8) -> dict[str, DataLoader]:
    train_aug, train_plain = Food10("train", food_transform(aug)), Food10("train", food_transform("none"))
    test = Food10("test", food_transform("none"))
    # 10 % of each class for validation; the file is sorted by class, so every 10th image is stratified
    val_idx = list(range(0, len(train_aug), 10))
    train_idx = sorted(set(range(len(train_aug))) - set(val_idx))
    return {
        "train": _loader(Subset(train_aug, train_idx), batch, True, workers),
        "val": _loader(Subset(train_plain, val_idx), 128, False, 4),
        "test": _loader(test, 128, False, 4),
    }
