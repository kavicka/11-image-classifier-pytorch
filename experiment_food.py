"""Transfer learning on a 10-class Food-101 subset, compared with training from scratch.

Stages:
  scratch    resnet18 with random weights: what 6,750 images give you without ImageNet
  backbones  3 pretrained backbones × (linear probe | fine-tune), compared on validation
  sweep      best backbone, fine-tune learning rate × augmentation, compared on validation
  final      best config, 3 seeds, evaluated once on test

Run:  uv run python experiment_food.py [--stage scratch|backbones|sweep|final|all]
"""

import argparse
import itertools
import json

import numpy as np
import timm
import torch

from common import MODELS, RESULTS, get_device, seed_all, setup_mlflow, update_results
from data import FOOD_CLASSES, food_loaders
from engine import Config, accuracy, confusion, fit, per_class, predict, top_confusions
from models_cnn import count_params

BACKBONES = ("resnet18.a1_in1k", "efficientnet_b0.ra_in1k", "convnext_tiny.fb_in22k_ft_in1k")
# one ConvNeXt run takes about 27 minutes on MPS, so the grid is 2 × 2 and reuses the backbone run
SWEEP_LRS = (1e-4, 3e-4)
SWEEP_AUGS = ("light", "strong")
PROBE_EPOCHS, FINETUNE_EPOCHS, SCRATCH_EPOCHS = 8, 10, 30
SEEDS = (0, 1, 2)


def freeze_backbone(model) -> tuple[list, callable]:
    """Train only the new head. Returns its parameters and a mode setter for the training loop."""
    for p in model.parameters():
        p.requires_grad = False
    head = model.get_classifier()
    for p in head.parameters():
        p.requires_grad = True

    def set_mode(m) -> None:
        # eval() keeps the backbone's BatchNorm on ImageNet statistics, otherwise it drifts despite frozen weights
        m.eval()
        head.train()

    return list(head.parameters()), set_mode


def param_groups(model, lr: float) -> list[dict]:
    """The random head learns 10× faster than the pretrained body, which only needs nudging."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    return [{"params": body, "lr": lr}, {"params": model.get_classifier().parameters(), "lr": lr * 10}]


def train_one(backbone: str, mode: str, cfg: Config, device) -> tuple[dict, torch.nn.Module, dict]:
    seed_all(cfg.seed)
    model = timm.create_model(backbone, pretrained=mode != "scratch", num_classes=len(FOOD_CLASSES))
    loaders = food_loaders(cfg.aug)
    if mode == "probe":
        params, set_mode = freeze_backbone(model)
        res = fit(model, loaders, cfg, device, params=params, set_mode=set_mode)
    elif mode == "finetune":
        res = fit(model, loaders, cfg, device, params=param_groups(model, cfg.lr))
    else:
        res = fit(model, loaders, cfg, device)
    res.update(backbone=backbone, mode=mode, params=count_params(model))
    return res, model, loaders


def scratch(device) -> None:
    cfg = Config("scratch-resnet18", lr=1e-3, epochs=SCRATCH_EPOCHS, aug="strong", weight_decay=0.05, optimizer="adamw")
    res, model, loaders = train_one("resnet18", "scratch", cfg, device)
    res["test_acc"] = accuracy(*predict(model, loaders["test"], device))
    print(f"scratch test accuracy {res['test_acc']:.4f}")
    update_results("food_scratch", res)


def backbones(device) -> None:
    runs = []
    for backbone in BACKBONES:
        short = backbone.split(".")[0]
        cfg = Config(f"probe-{short}", lr=1e-3, epochs=PROBE_EPOCHS, aug="light", weight_decay=1e-4, optimizer="adamw")
        runs.append(train_one(backbone, "probe", cfg, device)[0])
        cfg = Config(
            f"finetune-{short}", lr=3e-4, epochs=FINETUNE_EPOCHS, aug="light", weight_decay=0.01, optimizer="adamw"
        )
        runs.append(train_one(backbone, "finetune", cfg, device)[0])
    update_results("food_backbones", runs)


def sweep(device) -> None:
    saved = json.loads(RESULTS.read_text())
    best = max((r for r in saved["food_backbones"] if r["mode"] == "finetune"), key=lambda r: r["best_val_acc"])
    backbone = best["backbone"]
    short = backbone.split(".")[0]
    runs = saved.get("food_sweep") or [best]  # every finished run is saved, so an interrupted Mac can resume
    for lr, aug in itertools.product(SWEEP_LRS, SWEEP_AUGS):
        if any(r["lr"] == lr and r["aug"] == aug for r in runs):
            continue
        cfg = Config(
            f"sweep-{short}-lr{lr}-{aug}", lr=lr, epochs=FINETUNE_EPOCHS, aug=aug, weight_decay=0.01, optimizer="adamw"
        )
        runs.append(train_one(backbone, "finetune", cfg, device)[0])
        update_results("food_sweep", runs)


def final(device) -> None:
    saved = json.loads(RESULTS.read_text())
    best = max(saved["food_sweep"], key=lambda r: r["best_val_acc"])
    backbone, lr, aug = best["backbone"], best["lr"], best["aug"]
    print(f"best: {backbone} lr={lr} aug={aug} val={best['best_val_acc']:.4f}")
    out = saved.get("food_final_partial") or {
        "backbone": backbone,
        "lr": lr,
        "aug": aug,
        "epochs": FINETUNE_EPOCHS,
        "seeds": [],
    }
    best_val = max((r["best_val_acc"] for r in out["seeds"]), default=-1.0)
    for seed in SEEDS:
        if any(r["seed"] == seed for r in out["seeds"]):
            continue
        cfg = Config(
            f"final-seed{seed}", lr=lr, epochs=FINETUNE_EPOCHS, aug=aug, weight_decay=0.01, optimizer="adamw", seed=seed
        )
        res, model, loaders = train_one(backbone, "finetune", cfg, device)
        probs, y = predict(model, loaders["test"], device)
        res["test_acc"] = accuracy(probs, y)
        res["test_top3"] = float((np.argsort(-probs, 1)[:, :3] == y[:, None]).any(1).mean())
        out["seeds"].append(res)
        print(f"{cfg.name}: val {res['best_val_acc']:.4f} test {res['test_acc']:.4f}")
        if res["best_val_acc"] > best_val:  # chosen on validation
            best_val = res["best_val_acc"]
            MODELS.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODELS / "food10.pt")
            m = confusion(probs, y, len(FOOD_CLASSES))
            out["best_seed"] = seed
            out["confusion"] = m.tolist()
            out["per_class"] = per_class(m, FOOD_CLASSES)
            out["top_confusions"] = top_confusions(m, FOOD_CLASSES)
        update_results("food_final_partial", out)
    accs = [r["test_acc"] for r in out["seeds"]]
    out["test_mean"], out["test_std"] = float(np.mean(accs)), float(np.std(accs, ddof=1))
    print(f"food10 test {out['test_mean']:.4f} ± {out['test_std']:.4f}")
    update_results("food_final", out)
    update_results("food_final_partial", None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["scratch", "backbones", "sweep", "final", "all"])
    stage = ap.parse_args().stage
    device = get_device()
    print("device:", device)
    setup_mlflow("food10")
    if stage in ("scratch", "all"):
        scratch(device)
    if stage in ("backbones", "all"):
        backbones(device)
    if stage in ("sweep", "all"):
        sweep(device)
    if stage in ("final", "all"):
        final(device)
