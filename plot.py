"""Turn results.json into the plots used in the README (measurement and presentation stay separate)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import CIFAR_CLASSES, FOOD_CLASSES

R = json.loads(Path("results.json").read_text())
Path("assets").mkdir(exist_ok=True)
# the colours used across all steps
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"
AUG_COLOR = {"none": GRAY, "crop_flip": BLUE, "crop_flip_erase": GREEN, "light": BLUE, "strong": GREEN}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def sweep(runs: list[dict], title: str, name: str) -> None:
    fig, a = plt.subplots(figsize=(7, 4.2))
    for aug in dict.fromkeys(r["aug"] for r in runs):
        sub = sorted((r for r in runs if r["aug"] == aug), key=lambda r: r["lr"])
        a.plot([r["lr"] for r in sub], [r["best_val_acc"] for r in sub], "-o", color=AUG_COLOR[aug], label=aug)
    a.set(xscale="log", xlabel="peak learning rate", ylabel="best validation accuracy", title=title)
    a.grid(alpha=0.3)
    a.legend(title="augmentation")
    save(fig, name)


def curves(cfg: dict) -> None:
    fig, a = plt.subplots(figsize=(7, 4.2))
    run = next(r for r in cfg["resnet20"] if r["seed"] == cfg["best_seed"])
    small = next(r for r in cfg["small_cnn"] if r["seed"] == cfg["best_seed"])
    for r, color, label in ((run, BLUE, "ResNet-20"), (small, ORANGE, "small CNN")):
        e = [h["epoch"] + 1 for h in r["history"]]
        a.plot(e, [h["train_acc"] for h in r["history"]], color=color, ls="--", lw=1.2, label=f"{label} train")
        a.plot(e, [h["val_acc"] for h in r["history"]], color=color, lw=2, label=f"{label} val")
    a.set(xlabel="epoch", ylabel="accuracy", title="CIFAR-10: train vs validation", ylim=(0.3, 1.0))
    a.grid(alpha=0.3)
    a.legend()
    save(fig, "cifar_curves.png")


def confusion(m: list, classes, title: str, name: str) -> None:
    m = np.array(m)
    share = m / m.sum(1, keepdims=True)
    n = len(classes)
    fig, a = plt.subplots(figsize=(7.4, 6.4))
    a.imshow(share, cmap="Blues", vmin=0, vmax=1)
    labels = [c.replace("_", " ") for c in classes]
    a.set_xticks(range(n), labels, rotation=45, ha="right")
    a.set_yticks(range(n), labels)
    a.set(xlabel="predicted", ylabel="true", title=title)
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                a.text(
                    j, i, m[i, j], ha="center", va="center", fontsize=8, color="white" if share[i, j] > 0.5 else DARK
                )
    save(fig, name)


def transfer(res: dict) -> None:
    runs = res["food_backbones"]
    names = list(dict.fromkeys(r["backbone"].split(".")[0] for r in runs))
    x = np.arange(len(names))
    probe = [
        next(r for r in runs if r["backbone"].startswith(n) and r["mode"] == "probe")["best_val_acc"] for n in names
    ]
    fine = [
        next(r for r in runs if r["backbone"].startswith(n) and r["mode"] == "finetune")["best_val_acc"] for n in names
    ]
    fig, a = plt.subplots(figsize=(7, 4.2))
    a.bar(x - 0.2, probe, 0.4, color="#93c5fd", label="linear probe (frozen)")
    a.bar(x + 0.2, fine, 0.4, color=BLUE, label="fine-tune")
    a.axhline(res["food_scratch"]["best_val_acc"], color=RED, ls="--", label="resnet18 from scratch")
    a.axhline(0.9, color=GRAY, ls=":", label="goal 90 %")
    a.set_xticks(x, names)
    a.set(ylabel="best validation accuracy", ylim=(0, 1), title="Food-10: what pretrained weights buy")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, "food_transfer.png")


def main() -> None:
    sweep(R["cifar_sweep"], "CIFAR-10: ResNet-20, 15 epochs", "cifar_sweep.png")
    curves(R["cifar_final"])
    confusion(R["cifar_final"]["confusion"], CIFAR_CLASSES, "CIFAR-10 test: ResNet-20", "cifar_confusion.png")
    if "food_final" in R:
        transfer(R)
        sweep(R["food_sweep"], f"Food-10: {R['food_final']['backbone'].split('.')[0]}, fine-tuning", "food_sweep.png")
        confusion(R["food_final"]["confusion"], FOOD_CLASSES, "Food-10 test", "food_confusion.png")


if __name__ == "__main__":
    main()
