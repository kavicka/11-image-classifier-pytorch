"""The training loop, written out instead of hidden in a framework.

One step is always the same five lines: forward, loss, zero_grad, backward, step.
Everything else here is bookkeeping around them.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import mlflow
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class Config:
    name: str
    lr: float
    epochs: int
    aug: str
    weight_decay: float = 5e-4
    optimizer: str = "sgd"
    seed: int = 0


def train_epoch(model, loader, optimizer, scheduler, device, set_mode: Callable | None = None) -> tuple[float, float]:
    # train() turns on BatchNorm updates and dropout; a frozen backbone overrides that through set_mode
    set_mode(model) if set_mode else model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # forward
        loss = F.cross_entropy(logits, y)  # softmax + negative log-likelihood from step 08
        optimizer.zero_grad(set_to_none=True)  # gradients accumulate unless cleared
        loss.backward()  # autograd from step 04, at scale
        optimizer.step()  # w ← w − lr · (gradient + momentum + weight decay)
        scheduler.step()  # per batch, so the learning rate curve is smooth
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities and true labels. eval() makes BatchNorm use its running statistics."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean())


def make_optimizer(params, cfg: Config) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def fit(model, loaders, cfg: Config, device, params=None, set_mode=None) -> dict:
    """Train with a one-cycle learning rate, log every epoch to MLflow, keep the best weights on val."""
    model.to(device)
    optimizer = make_optimizer(model.parameters() if params is None else params, cfg)
    max_lr = [g["lr"] for g in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=cfg.epochs * len(loaders["train"]), pct_start=0.1
    )
    history, best_val, best_state = [], -1.0, None
    t0 = time.perf_counter()
    with mlflow.start_run(run_name=cfg.name):
        mlflow.log_params(asdict(cfg))
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch(model, loaders["train"], optimizer, scheduler, device, set_mode)
            val_acc = accuracy(*predict(model, loaders["val"], device))
            history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
            mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc}, step=epoch)
            print(
                f"{cfg.name} {epoch + 1}/{cfg.epochs} loss {train_loss:.3f} train {train_acc:.3f} val {val_acc:.3f}",
                flush=True,
            )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        seconds = time.perf_counter() - t0
        mlflow.log_metrics({"best_val_acc": best_val, "seconds": seconds})
    model.load_state_dict(best_state)  # early stopping on val, without stopping
    return {**asdict(cfg), "best_val_acc": best_val, "seconds": seconds, "history": history}


def confusion(probs: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), int)
    np.add.at(m, (y, probs.argmax(1)), 1)
    return m


def top_confusions(m: np.ndarray, classes, k: int = 8) -> list[dict]:
    off = m.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [{"true": classes[i], "pred": classes[j], "count": int(off[i, j])} for i, j in pairs]


def per_class(m: np.ndarray, classes) -> dict[str, float]:
    return {c: float(m[i, i] / m[i].sum()) for i, c in enumerate(classes)}
