"""CIFAR-10 with models written by hand and my own training loop.

Stages:
  baselines  majority class, linear model on raw pixels
  sweep      ResNet-20, learning rate × augmentation, short runs, compared on validation
  final      best sweep config, long runs, 3 seeds, evaluated once on test

Run:  uv run python experiment_cifar.py [--stage baselines|sweep|final|all]
View: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import itertools
import json

import numpy as np
import torch

from common import MODELS, RESULTS, get_device, seed_all, setup_mlflow, update_results
from data import CIFAR_CLASSES, cifar_loaders
from engine import Config, accuracy, confusion, fit, per_class, predict, top_confusions
from models_cnn import Linear, ResNet, SmallCNN, count_params

SWEEP_LRS = (0.003, 0.01, 0.03, 0.1, 0.3)
SWEEP_AUGS = ("none", "crop_flip", "crop_flip_erase")
SWEEP_EPOCHS = 15
FINAL_EPOCHS = 60
SEEDS = (0, 1, 2)


def run(model, cfg: Config, device) -> tuple[dict, dict]:
    seed_all(cfg.seed)
    loaders = cifar_loaders(cfg.aug)
    res = fit(model, loaders, cfg, device)
    return res, loaders


def baselines(device) -> None:
    majority = 0.1  # CIFAR-10 test set has exactly 1000 images per class
    cfg = Config("linear", lr=0.01, epochs=15, aug="none")
    model = Linear()
    res, loaders = run(model, cfg, device)
    test = accuracy(*predict(model, loaders["test"], device))
    print(f"linear test accuracy {test:.4f}")
    update_results("cifar_baselines", {"majority_acc": majority, "linear_test_acc": test, "linear": res})


def sweep(device) -> None:
    runs = []
    for lr, aug in itertools.product(SWEEP_LRS, SWEEP_AUGS):
        cfg = Config(f"resnet20-lr{lr}-{aug}", lr=lr, epochs=SWEEP_EPOCHS, aug=aug)
        runs.append(run(ResNet(n=3), cfg, device)[0])
    update_results("cifar_sweep", runs)


def final(device) -> None:
    sweep_runs = json.loads(RESULTS.read_text())["cifar_sweep"]
    best = max(sweep_runs, key=lambda r: r["best_val_acc"])
    lr, aug = best["lr"], best["aug"]
    print(f"best sweep config: lr={lr} aug={aug} val={best['best_val_acc']:.4f}")
    # long stage: every finished run is saved, so an interrupted Mac picks up where it stopped
    saved = json.loads(RESULTS.read_text()).get("cifar_final_partial")
    out = saved or {"lr": lr, "aug": aug, "epochs": FINAL_EPOCHS, "resnet20": [], "small_cnn": []}
    best_val = max((r["best_val_acc"] for r in out["resnet20"]), default=-1.0)
    for seed in SEEDS:
        for key, make in (("resnet20", lambda: ResNet(n=3)), ("small_cnn", SmallCNN)):
            if any(r["seed"] == seed for r in out[key]):
                continue
            model = make()
            cfg = Config(f"final-{key}-seed{seed}", lr=lr, epochs=FINAL_EPOCHS, aug=aug, seed=seed)
            res, loaders = run(model, cfg, device)
            probs, y = predict(model, loaders["test"], device)
            res.update(test_acc=accuracy(probs, y), params=count_params(model))
            out[key].append(res)
            print(f"{cfg.name}: val {res['best_val_acc']:.4f} test {res['test_acc']:.4f} {res['seconds']:.0f} s")
            # the model that goes on is chosen on validation, never on test
            if key == "resnet20" and res["best_val_acc"] > best_val:
                best_val = res["best_val_acc"]
                MODELS.mkdir(exist_ok=True)
                torch.save(model.state_dict(), MODELS / "resnet20.pt")
                m = confusion(probs, y, 10)
                out["best_seed"] = seed
                out["confusion"] = m.tolist()
                out["per_class"] = per_class(m, CIFAR_CLASSES)
                out["top_confusions"] = top_confusions(m, CIFAR_CLASSES)
            update_results("cifar_final_partial", out)
    for key in ("resnet20", "small_cnn"):
        accs = [r["test_acc"] for r in out[key]]
        out[f"{key}_test_mean"], out[f"{key}_test_std"] = float(np.mean(accs)), float(np.std(accs, ddof=1))
        print(f"{key}: test {out[f'{key}_test_mean']:.4f} ± {out[f'{key}_test_std']:.4f}")
    update_results("cifar_final", out)
    update_results("cifar_final_partial", None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["baselines", "sweep", "final", "all"])
    stage = ap.parse_args().stage
    device = get_device()
    print("device:", device)
    setup_mlflow("cifar10")
    if stage in ("baselines", "all"):
        baselines(device)
    if stage in ("sweep", "all"):
        sweep(device)
    if stage in ("final", "all"):
        final(device)
