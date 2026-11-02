# Tutorial · 11 · P3: Image classifier in PyTorch

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial rebuilds the project file by file in the order you would write it: data → training loop → models → CIFAR-10 experiments → transfer learning on food photos → Grad-CAM → ONNX export. Each step explains **what** the code does, **why** it is written that way and **how** it works. The README has all the numbers.

**Goal:** a CNN trained with your own PyTorch loop on the Mac GPU (MPS): ≥ 80 % on CIFAR-10 from scratch, ≥ 90 % on 10 food classes with transfer learning, every experiment tracked in MLflow.

**You need:** projects 03 (shapes), 04 (training loop, autograd) and 08 (cross-entropy). Libraries: PyTorch + torchvision, timm (pretrained models), MLflow, ONNX Runtime.

---

## Step 1 · Shared helpers ([`common.py`](common.py))

```python
def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def setup_mlflow(experiment):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

- **Device detection** from project 01: MPS on the Mac, CPU elsewhere.
- **Seeding** all three random generators so runs are repeatable. MPS is not bit-for-bit deterministic, so repeats differ slightly; that is why final models use 3 seeds.
- **MLflow in one SQLite file** instead of thousands of small files. `update_results(key, value)` lets each stage own one key in `results.json`, so rerunning one stage doesn't erase the others.

---

## Step 2 · Data, splits and augmentation ([`data.py`](data.py))

### CIFAR-10

```python
idx = torch.randperm(50_000, generator=torch.Generator().manual_seed(0)).tolist()
train = Subset(train_aug,   idx[5_000:])    # 45,000 with augmentation
val   = Subset(train_plain, idx[:5_000])    # 5,000 without augmentation
test  = CIFAR10(train=False)                # 10,000, touched only by final models
```

**Why a validation split carved from training:** hyperparameters are chosen on validation. If you chose them on the test set, the final test number would be optimistic. The split uses a fixed seed so every run sees the same validation images.

**Why two copies of the training set:** validation images must never be augmented; they must look like test images.

**Augmentation** (`cifar_transform`):
```python
"crop_flip": [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
```
Padding by 4 and cropping back to 32 shifts the image by up to 4 pixels; flipping mirrors it. A shifted or mirrored cat is still a cat, so the model gets "new" examples for free and learns not to depend on exact positions. Random erasing (hide a patch) is also available; it is applied **after** normalisation.

**Normalisation** with the dataset's per-channel mean and std centres inputs around 0, which keeps early activations in a sensible range.

### Food-10

The first 10 Food-101 classes **alphabetically** (apple pie … breakfast burrito). **Why alphabetically:** nobody can say the easy dishes were picked. `ensure_food10()` downloads the 5 GB archive once, unpacks only those 10 classes and deletes the archive. Training images get a random resized crop to 224×224 plus a flip; validation and test images are resized to 256 and centre-cropped to 224. All are normalised with **ImageNet** statistics, because that is what the pretrained weights expect.

---

## Step 3 · The training loop ([`engine.py`](engine.py))

```python
def train_epoch(model, loader, optimizer, scheduler, device, set_mode=None):
    set_mode(model) if set_mode else model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                          # forward
        loss = F.cross_entropy(logits, y)          # project 08
        optimizer.zero_grad(set_to_none=True)      # gradients accumulate otherwise
        loss.backward()                            # project 04, at scale
        optimizer.step()
        scheduler.step()                           # per batch
```

**Why written out instead of using a framework:** these five lines are the same loop as project 04's micrograd, now on tensors and a GPU. Seeing them keeps nothing magic.

**`model.train()` vs `model.eval()`:** BatchNorm normalises with the *current batch's* statistics in training and with *running averages* in evaluation. Forgetting `eval()` makes predictions depend on what else is in the batch. `predict()` always calls `model.eval()` and `@torch.no_grad()` (no graph needed, saves memory).

**Optimizer** (`make_optimizer`): SGD with Nesterov momentum 0.9 and weight decay 5e-4 for CIFAR (the classic recipe for ResNets); AdamW for all Food-10 runs (pretrained models and the from-scratch baseline).

**One-cycle learning rate** (`fit`):
```python
OneCycleLR(optimizer, max_lr=..., total_steps=epochs * len(train_loader), pct_start=0.1)
```
The rate rises quickly for the first 10 % of steps, then decays to near zero. **Why:** high rates early move weights far and act as regularisation; low rates late let them settle into a good minimum. Updated every batch so the curve is smooth.

**Keep the best weights on validation:**
```python
if val_acc > best_val:
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
...
model.load_state_dict(best_state)
```
Early stopping without stopping: training runs all epochs, but the returned model is the one from the best validation epoch.

**MLflow:** every run logs its config (`log_params`) and per-epoch metrics, so 37 runs can be compared in `mlflow ui`.

---

## Step 4 · Three models of increasing power ([`models_cnn.py`](models_cnn.py))

### Linear baseline
One matrix from raw pixels to 10 scores. Every pixel has its own weight per class; it knows nothing about neighbouring pixels. **40.04 %**: the floor any CNN must beat.

### SmallCNN
```python
conv_block(3, 32), conv_block(32, 32), MaxPool2d(2),     # (B, 32, 16, 16)
conv_block(32, 64), conv_block(64, 64), MaxPool2d(2),    # (B, 64, 8, 8)
conv_block(64, 128), conv_block(128, 128), MaxPool2d(2), # (B, 128, 4, 4)
head(x.mean((2, 3)))                                     # global average pool → (B, 128) → (B, 10)
```

- **Convolution:** a small 3×3 filter slides over the image. The same weights are used everywhere, so a feature learned in one corner (an edge, an eye) works in every corner, with far fewer weights than a dense layer.
- **`bias=False` before BatchNorm:** BatchNorm subtracts the mean anyway, so a bias there would be cancelled.
- **MaxPool** halves height and width, so deeper layers see larger regions; channels grow 32 → 64 → 128 to hold more kinds of features.
- **Global average pooling:** one number per channel ("how much of this feature is anywhere in the image"). For classification, *whether* something is present matters, not *where*.

### ResNet-20 with residual blocks
```python
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return F.relu(out + self.shortcut(x))       # F(x) + x
```

**Why residual connections:** stacking more plain layers eventually makes training *worse*, because the signal and gradients get distorted through many layers. A residual block only learns a correction `F(x)` added to its input. Doing nothing (F = 0) is easy, and gradients get a direct path backwards through the `+ x`.

**The shortcut:** when a block halves the resolution or changes channels, `x` no longer has the same shape as `F(x)`. A 1×1 convolution with stride 2 makes it addable.

**Structure:** 6n + 2 layers; n = 3 → ResNet-20 with three stages (16, 32, 64 channels at 32×32, 16×16, 8×8), 272,474 weights.

---

## Step 5 · CIFAR-10 experiments ([`experiment_cifar.py`](experiment_cifar.py))

1. **Baselines:** majority class (10 %) and the linear model (40.04 %).
2. **Sweep (15 runs, 15 epochs each):** peak learning rate × augmentation (none, flip, crop+flip, crop+flip+erase). Scored on validation.
3. **Final models:** the best config, 60 epochs, seeds 0, 1, 2, for SmallCNN and ResNet-20. Only these touch the test set.

**What the sweep taught:** the learning rate moved validation accuracy from 77.4 % to 88.4 %; augmentation only 1–3 points; random erasing added nothing beyond crop+flip. **Tune the learning rate first.**

**Results:** ResNet-20 **91.35 % ± 0.31**, SmallCNN 90.13 %. At this depth, residual connections give only ~1.2 points; they matter more as networks grow.

---

## Step 6 · Transfer learning on Food-10 ([`experiment_food.py`](experiment_food.py))

```python
model = timm.create_model(backbone, pretrained=mode != "scratch", num_classes=10)
```

`timm` downloads a network pretrained on ImageNet and replaces its last layer with a new 10-class head. Three backbones (ResNet-18, EfficientNet-B0, ConvNeXt-Tiny), three modes:

**From scratch** (ResNet-18, 30 epochs): 75.08 % test. 6,750 training images are not enough to learn vision from zero.

**Linear probe** (`freeze_backbone`):
```python
for p in model.parameters(): p.requires_grad = False
for p in head.parameters(): p.requires_grad = True
def set_mode(m):
    m.eval()      # keep BatchNorm on ImageNet statistics
    head.train()
```
Only the head trains. **Why `eval()` on the frozen backbone:** even with frozen weights, BatchNorm in `train()` mode would update its running statistics from food batches and drift. ConvNeXt probe: 91.87 % validation in 6 minutes.

**Fine-tuning** (`param_groups`):
```python
[{"params": body, "lr": lr}, {"params": head, "lr": lr * 10}]
```
Everything trains, but the random head learns 10× faster than the pretrained body, which only needs nudging. Too high a rate on the body would destroy what it learned on ImageNet.

**Final:** ConvNeXt-Tiny fine-tuned, lr 1e-4, strong augmentation, 3 seeds: **94.76 % ± 0.68** test. Lesson: **the choice of pretrained backbone mattered more than how you train it**; ConvNeXt's frozen features beat the other two fine-tuned.

---

## Step 7 · Error analysis with confusion matrices (`confusion`, `top_confusions`)

```python
m = np.zeros((n, n), int)
np.add.at(m, (y, probs.argmax(1)), 1)     # row = true class, column = predicted
```

`np.add.at` handles repeated (true, pred) pairs correctly (a plain `m[y, pred] += 1` would count each pair only once). The off-diagonal maximum shows which classes are confused: cat ↔ dog is 176 of 861 CIFAR errors; apple pie ↔ bread pudding 44 of 115 Food-10 errors. **One accuracy number hides that one pair causes 38 % of the errors.**

---

## Step 8 · Grad-CAM: where did the model look? ([`gradcam.py`](gradcam.py))

```python
handle = layer.register_forward_hook(lambda m, i, out: store.update(a=out))
score = model(x)[0, cls]
grads = torch.autograd.grad(score, store["a"])[0]
cam = F.relu((grads.mean((2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
cam = F.interpolate(cam, x.shape[2:], mode="bilinear")
```

**How it works:**
1. A **forward hook** saves the last convolutional feature maps `A` without changing the model.
2. `torch.autograd.grad` gives `∂score/∂A`: how much each cell of each channel would change the class score.
3. **Average the gradient over the map** → one weight per channel ("how important is this channel for this class").
4. **Weighted sum of the channels** → one heatmap; **ReLU** keeps only regions that speak *for* the class.
5. Upsample to image size.

**Why averaging is valid here:** these classifiers end in global average pooling, which is exactly the case Grad-CAM was derived for. (Racing Lab ch. 11 shows what goes wrong when the head flattens instead.)

**What it showed:** on CIFAR the maps cover the object even for mistakes. On Food-10 only 22.6 % of the heatmap falls in the central quarter (uniform would be 25 %): food is recognised by texture spread over the frame. Treat these maps as hints, not proof.

---

## Step 9 · Export to ONNX ([`export_onnx.py`](export_onnx.py))

```python
torch.onnx.export(model, (dummy,), "food10.onnx",
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
```

**Why ONNX:** a portable file that runs without PyTorch (ONNX Runtime, mobile, other languages). `dynamic_axes` allows any batch size.

**How it is verified:** run the whole test set through both PyTorch and ONNX Runtime: same accuracy (95.40 %), max logit difference 2.8e-5. **Latency** (median of 100, measured *before* the data loader starts so its worker processes don't compete for CPU): 299 ms PyTorch vs 15 ms ONNX Runtime on CPU.

---

## Step 10 · Run everything, safely

```bash
nohup caffeinate -i sh run_all.sh > train.log 2>&1 &   # ~5 hours on an M4 Pro
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Why detached and saving after each run:** a training run once died with its terminal session and lost an hour. Now every stage saves its results as it finishes, and `caffeinate` keeps the Mac awake.

---

## Try it yourself

1. Extend the CIFAR learning-rate grid above 0.3. The best result sat at the top of the grid; is the optimum higher?
2. Fine-tune ConvNeXt with the same learning rate for body and head (remove the ×10). How much accuracy do you lose?
3. Remove the `m.eval()` in `freeze_backbone` and rerun the linear probe. How much does BatchNorm drift cost?

## Recap

| Step | Idea |
|---|---|
| 2 | Validation split for choices, test only for final models; augment only training |
| 3 | Five-line loop; train/eval modes for BatchNorm; one-cycle LR; keep best-val weights |
| 4 | Convolution = shared local filters; residual block = learn a correction F(x) + x |
| 5 | Tune the learning rate first; augmentation is second |
| 6 | Pretrained features beat training from scratch on small data; head learns faster than body |
| 7 | Confusion matrix shows which pair of classes causes the errors |
| 8 | Grad-CAM = gradient-weighted feature maps; a hint, not proof |
| 9 | ONNX: same accuracy, portable, faster on CPU |
