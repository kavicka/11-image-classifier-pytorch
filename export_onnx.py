"""Export the Food-10 model to ONNX and prove it is the same model.

Checks: outputs close to PyTorch on real images, the same test accuracy, CPU latency for one image.

Run:  uv run python export_onnx.py   → models/food10.onnx, key "onnx" in results.json
"""

import json
import time

import numpy as np
import onnxruntime as ort
import timm
import torch

from common import MODELS, RESULTS, update_results
from data import FOOD_CLASSES, food_loaders
from engine import accuracy


def latency_ms(fn, x, repeats: int = 100) -> float:
    fn(x)  # first call allocates; not representative
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000)


def main() -> None:
    cfg = json.loads(RESULTS.read_text())["food_final"]
    model = timm.create_model(cfg["backbone"], pretrained=False, num_classes=len(FOOD_CLASSES))
    model.load_state_dict(torch.load(MODELS / "food10.pt"))
    model.eval()

    path = MODELS / "food10.onnx"
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    # measured before the data loader starts, because its worker processes would compete for CPU cores
    one = np.random.default_rng(0).normal(size=(1, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        pt_ms = latency_ms(lambda v: model(torch.from_numpy(v)), one)
    ox_ms = latency_ms(lambda v: session.run(None, {"image": v}), one)

    all_torch, all_onnx, labels = [], [], []
    for x, y in food_loaders("none")["test"]:
        with torch.no_grad():
            all_torch.append(model(x).numpy())
        all_onnx.append(session.run(None, {"image": x.numpy()})[0])
        labels.append(y.numpy())
    pt, ox, y = np.concatenate(all_torch), np.concatenate(all_onnx), np.concatenate(labels)

    res = {
        "max_abs_logit_diff": float(np.abs(pt - ox).max()),
        "torch_test_acc": accuracy(pt, y),
        "onnx_test_acc": accuracy(ox, y),
        "torch_cpu_ms": pt_ms,
        "onnx_cpu_ms": ox_ms,
        "onnx_mb": path.stat().st_size / 1e6,
    }
    print(res)
    update_results("onnx", res)


if __name__ == "__main__":
    main()
