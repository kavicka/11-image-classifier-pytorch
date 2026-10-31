"""Grad-CAM written by hand: where in the image did the model find its evidence?

For a class c: take the last convolutional feature maps A_k and the gradient of the score of c with
respect to them. The gradient averaged over the map says how important channel k is for c, so
  heatmap = ReLU(Σ_k mean(∂score_c/∂A_k) · A_k).
ReLU keeps only the regions that speak FOR the class.

Run:  uv run python gradcam.py   → assets/gradcam_food.png, assets/gradcam_cifar.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from torch.nn import functional as F

from common import MODELS, RESULTS, get_device, update_results
from data import (
    CIFAR_CLASSES,
    CIFAR_MEAN,
    CIFAR_STD,
    FOOD_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    cifar_loaders,
    food_loaders,
)
from engine import predict
from models_cnn import ResNet


def grad_cam(model, layer, x: torch.Tensor, cls: int) -> np.ndarray:
    """x is one image (1, C, H, W). Returns a heatmap in [0, 1] of the same height and width."""
    store = {}
    handle = layer.register_forward_hook(lambda m, i, out: store.update(a=out))
    score = model(x)[0, cls]
    grads = torch.autograd.grad(score, store["a"])[0]
    handle.remove()
    # convnext stages can use (B, C, H, W) as well; all three families here keep channels second
    cam = F.relu((grads.mean((2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu()
    return (cam / (cam.max() + 1e-8)).numpy()


def to_image(x: torch.Tensor, mean, std) -> np.ndarray:
    m, s = torch.tensor(mean)[:, None, None], torch.tensor(std)[:, None, None]
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def pick(probs: np.ndarray, y: np.ndarray, n: int) -> tuple[list[int], list[int]]:
    """The most confident hits and misses: an honest sample of the best and the worst moments."""
    pred, conf = probs.argmax(1), probs.max(1)
    hits = np.flatnonzero(pred == y)
    misses = np.flatnonzero(pred != y)
    return hits[np.argsort(-conf[hits])[:: len(hits) // n][:n]].tolist(), misses[np.argsort(-conf[misses])[:n]].tolist()


def figure(model, layer, ds, probs, y, classes, mean, std, device, path: str, n: int = 6, pretty=str) -> None:
    hits, misses = pick(probs, y, n)
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9.4))
    for row, idxs in ((0, hits), (2, misses)):
        for col, i in enumerate(idxs):
            x = ds[i][0][None].to(device)
            pred = int(probs[i].argmax())
            cam = grad_cam(model, layer, x, pred)
            img = to_image(x[0], mean, std)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f"true {pretty(classes[y[i]])}\npred {pretty(classes[pred])}", fontsize=8)
            axes[row + 1, col].imshow(img)
            axes[row + 1, col].imshow(cam, cmap="jet", alpha=0.45)
    for a in axes.ravel():
        a.set(xticks=[], yticks=[])
    axes[0, 0].set_ylabel("correct")
    axes[2, 0].set_ylabel("wrong")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def center_mass(model, layer, ds, y, probs, device, n: int = 200) -> float:
    """Share of the heatmap inside the central half of the image, against 25 % for a uniform map."""
    h = None
    total, inner = 0.0, 0.0
    for i in np.linspace(0, len(y) - 1, n).astype(int):
        cam = grad_cam(model, layer, ds[i][0][None].to(device), int(probs[i].argmax()))
        h = cam.shape[0]
        lo, hi = h // 4, h - h // 4
        total += cam.sum()
        inner += cam[lo:hi, lo:hi].sum()
    return float(inner / total)


def main() -> None:
    device = get_device()
    res = json.loads(RESULTS.read_text())
    food_cfg = res["food_final"]
    model = timm.create_model(food_cfg["backbone"], pretrained=False, num_classes=len(FOOD_CLASSES))
    model.load_state_dict(torch.load(MODELS / "food10.pt"))
    model.to(device).eval()
    loaders = food_loaders("none")
    probs, y = predict(model, loaders["test"], device)
    layer = model.stages[-1] if hasattr(model, "stages") else model.layer4
    pretty = lambda s: s.replace("_", " ")
    ds = loaders["test"].dataset
    figure(
        model,
        layer,
        ds,
        probs,
        y,
        FOOD_CLASSES,
        IMAGENET_MEAN,
        IMAGENET_STD,
        device,
        "assets/gradcam_food.png",
        pretty=pretty,
    )
    food_center = center_mass(model, layer, ds, y, probs, device)

    net = ResNet(n=3)
    net.load_state_dict(torch.load(MODELS / "resnet20.pt"))
    net.to(device).eval()
    loaders = cifar_loaders("none", workers=2)
    probs, y = predict(net, loaders["test"], device)
    ds = loaders["test"].dataset
    figure(net, net.stage3, ds, probs, y, CIFAR_CLASSES, CIFAR_MEAN, CIFAR_STD, device, "assets/gradcam_cifar.png")
    update_results("gradcam", {"food_center_share": food_center, "uniform_center_share": 0.25})
    print("food heatmap share in central half:", food_center)


if __name__ == "__main__":
    main()
