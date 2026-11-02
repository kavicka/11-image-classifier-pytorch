# Tutoriál · 11 · P3: Klasifikátor obrázkov v PyTorchi

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál postaví projekt súbor po súbore v poradí, v akom by si ho písal: dáta → tréningový cyklus → modely → experimenty na CIFAR-10 → transfer learning na fotkách jedla → Grad-CAM → export do ONNX. Každý krok vysvetľuje, **čo** kód robí, **prečo** je napísaný takto a **ako** funguje. README má všetky čísla.

**Cieľ:** CNN natrénovaná vlastným cyklom v PyTorchi na GPU Macu (MPS): ≥ 80 % na CIFAR-10 od nuly, ≥ 90 % na 10 triedach jedla s transfer learningom, každý experiment zaznamenaný v MLflow.

**Potrebuješ:** projekty 03 (tvary), 04 (tréningový cyklus, autograd) a 08 (cross-entropy). Knižnice: PyTorch + torchvision, timm (predtrénované modely), MLflow, ONNX Runtime.

---

## Krok 1 · Spoločné pomocné funkcie ([`common.py`](common.py))

```python
def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def setup_mlflow(experiment):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

- **Detekcia zariadenia** z projektu 01: MPS na Macu, inde CPU.
- **Seedovanie** všetkých troch generátorov náhody, aby sa behy dali zopakovať. MPS nie je deterministické bit po bite, takže opakovania sa mierne líšia; preto finálne modely používajú 3 seedy.
- **MLflow v jednom SQLite súbore** namiesto tisícov malých súborov. `update_results(key, value)` dovolí každej fáze vlastniť jeden kľúč v `results.json`, takže opätovné spustenie jednej fázy nezmaže ostatné.

---

## Krok 2 · Dáta, rozdelenie a augmentácia ([`data.py`](data.py))

### CIFAR-10

```python
idx = torch.randperm(50_000, generator=torch.Generator().manual_seed(0)).tolist()
train = Subset(train_aug,   idx[5_000:])    # 45 000 s augmentáciou
val   = Subset(train_plain, idx[:5_000])    # 5 000 bez augmentácie
test  = CIFAR10(train=False)                # 10 000, siahnu naň len finálne modely
```

**Prečo validačná časť vyrezaná z trénovacej:** hyperparametre sa vyberajú na validácii. Keby si ich vyberal na teste, finálne číslo na teste by bolo optimistické. Rozdelenie používa pevný seed, takže každý beh vidí rovnaké validačné obrázky.

**Prečo dve kópie trénovacej sady:** validačné obrázky sa nikdy nesmú augmentovať; musia vyzerať ako testovacie.

**Augmentácia** (`cifar_transform`):
```python
"crop_flip": [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
```
Doplnenie o 4 pixely a orezanie späť na 32 posunie obrázok až o 4 pixely; flip ho zrkadlí. Posunutá alebo zrkadlená mačka je stále mačka, takže model dostane „nové“ príklady zadarmo a naučí sa nezávisieť od presnej polohy. K dispozícii je aj random erasing (skrytie kúska); aplikuje sa **po** normalizácii.

**Normalizácia** priemerom a smerodajnou odchýlkou datasetu pre každý kanál vycentruje vstupy okolo 0, čo drží prvé aktivácie v rozumnom rozsahu.

### Food-10

Prvých 10 tried Food-101 **podľa abecedy** (apple pie … breakfast burrito). **Prečo podľa abecedy:** nikto nemôže povedať, že sa vybrali ľahké jedlá. `ensure_food10()` raz stiahne 5 GB archív, rozbalí len tých 10 tried a archív zmaže. Trénovacie obrázky dostanú náhodný orez so zmenou veľkosti na 224×224 a flip; validačné a testovacie sa zmenšia na 256 a orežú zo stredu na 224. Všetky sa normalizujú štatistikami **ImageNetu**, lebo tie očakávajú predtrénované váhy.

---

## Krok 3 · Tréningový cyklus ([`engine.py`](engine.py))

```python
def train_epoch(model, loader, optimizer, scheduler, device, set_mode=None):
    set_mode(model) if set_mode else model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                          # forward
        loss = F.cross_entropy(logits, y)          # projekt 08
        optimizer.zero_grad(set_to_none=True)      # inak sa gradienty sčítavajú
        loss.backward()                            # projekt 04, vo veľkom
        optimizer.step()
        scheduler.step()                           # po každom batchi
```

**Prečo vypísané a nie cez framework:** týchto päť riadkov je ten istý cyklus ako micrograd v projekte 04, teraz na tenzoroch a GPU. Keď ich vidíš, nič nie je mágia.

**`model.train()` vs `model.eval()`:** BatchNorm pri tréningu normalizuje štatistikami *aktuálneho batchu* a pri vyhodnotení *kĺzavými priemermi*. Keď zabudneš `eval()`, predikcie závisia od toho, čo ešte je v batchi. `predict()` vždy volá `model.eval()` a `@torch.no_grad()` (graf netreba, šetrí pamäť).

**Optimalizátor** (`make_optimizer`): SGD s Nesterovovým momentom 0,9 a weight decay 5e-4 pre CIFAR (klasický recept pre ResNety); AdamW pre všetky behy na Food-10 (predtrénované modely aj baseline od nuly).

**One-cycle learning rate** (`fit`):
```python
OneCycleLR(optimizer, max_lr=..., total_steps=epochs * len(train_loader), pct_start=0.1)
```
Rate rýchlo stúpne počas prvých 10 % krokov, potom klesá takmer na nulu. **Prečo:** vysoké hodnoty na začiatku posunú váhy ďaleko a pôsobia ako regularizácia; nízke na konci ich nechajú usadiť sa v dobrom minime. Aktualizuje sa po každom batchi, aby bola krivka hladká.

**Ponechaj najlepšie váhy podľa validácie:**
```python
if val_acc > best_val:
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
...
model.load_state_dict(best_state)
```
Early stopping bez zastavenia: tréning prebehne všetky epochy, ale vrátený model je z epochy s najlepšou validáciou.

**MLflow:** každý beh zaznamená konfiguráciu (`log_params`) a metriky po epochách, takže 37 behov sa dá porovnať v `mlflow ui`.

---

## Krok 4 · Tri modely so stúpajúcou silou ([`models_cnn.py`](models_cnn.py))

### Lineárny baseline
Jedna matica zo surových pixelov na 10 skóre. Každý pixel má vlastnú váhu pre každú triedu; o susedných pixeloch nevie nič. **40,04 %**: dno, ktoré musí každá CNN poraziť.

### SmallCNN
```python
conv_block(3, 32), conv_block(32, 32), MaxPool2d(2),     # (B, 32, 16, 16)
conv_block(32, 64), conv_block(64, 64), MaxPool2d(2),    # (B, 64, 8, 8)
conv_block(64, 128), conv_block(128, 128), MaxPool2d(2), # (B, 128, 4, 4)
head(x.mean((2, 3)))                                     # global average pool → (B, 128) → (B, 10)
```

- **Konvolúcia:** malý filter 3×3 sa posúva po obrázku. Všade sa používajú tie isté váhy, takže príznak naučený v jednom rohu (hrana, oko) funguje v každom rohu, s oveľa menej váhami než hustá vrstva.
- **`bias=False` pred BatchNorm:** BatchNorm aj tak odčíta priemer, takže bias by sa tam vyrušil.
- **MaxPool** zmenší výšku a šírku na polovicu, takže hlbšie vrstvy vidia väčšie oblasti; kanálov pribúda 32 → 64 → 128, aby uniesli viac druhov príznakov.
- **Global average pooling:** jedno číslo na kanál („koľko tohto príznaku je kdekoľvek na obrázku“). Pri klasifikácii záleží na tom, *či* tam niečo je, nie *kde*.

### ResNet-20 s reziduálnymi blokmi
```python
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    return F.relu(out + self.shortcut(x))       # F(x) + x
```

**Prečo reziduálne spojenia:** keď ukladáš na seba viac obyčajných vrstiev, tréning sa časom *zhorší*, lebo signál aj gradienty sa cez veľa vrstiev skreslia. Reziduálny blok sa učí len korekciu `F(x)` pripočítanú k vstupu. Nerobiť nič (F = 0) je ľahké a gradienty majú priamu cestu dozadu cez `+ x`.

**Skratka (shortcut):** keď blok zmenší rozlíšenie na polovicu alebo zmení počet kanálov, `x` už nemá rovnaký tvar ako `F(x)`. Konvolúcia 1×1 so stride 2 ho spraví sčítateľným.

**Štruktúra:** 6n + 2 vrstiev; n = 3 → ResNet-20 s tromi fázami (16, 32, 64 kanálov pri 32×32, 16×16, 8×8), 272 474 váh.

---

## Krok 5 · Experimenty na CIFAR-10 ([`experiment_cifar.py`](experiment_cifar.py))

1. **Baseliny:** najčastejšia trieda (10 %) a lineárny model (40,04 %).
2. **Sweep (15 behov po 15 epoch):** maximálny learning rate × augmentácia (žiadna, flip, crop+flip, crop+flip+erase). Hodnotí sa na validácii.
3. **Finálne modely:** najlepšia konfigurácia, 60 epoch, seedy 0, 1, 2, pre SmallCNN aj ResNet-20. Len tieto sa dotknú testu.

**Čo ukázal sweep:** learning rate pohol validačnú presnosť zo 77,4 % na 88,4 %; augmentácia len o 1–3 body; random erasing nepridal nič navyše ku crop+flip. **Najprv lad learning rate.**

**Výsledky:** ResNet-20 **91,35 % ± 0,31**, SmallCNN 90,13 %. Pri tejto hĺbke dajú reziduálne spojenia len ~1,2 bodu; dôležitejšie sú pri väčších sieťach.

---

## Krok 6 · Transfer learning na Food-10 ([`experiment_food.py`](experiment_food.py))

```python
model = timm.create_model(backbone, pretrained=mode != "scratch", num_classes=10)
```

`timm` stiahne sieť predtrénovanú na ImageNete a vymení jej poslednú vrstvu za novú hlavu s 10 triedami. Tri backbony (ResNet-18, EfficientNet-B0, ConvNeXt-Tiny), tri režimy:

**Od nuly** (ResNet-18, 30 epoch): 75,08 % na teste. 6 750 trénovacích obrázkov nestačí na to, aby sa model naučil vidieť od nuly.

**Linear probe** (`freeze_backbone`):
```python
for p in model.parameters(): p.requires_grad = False
for p in head.parameters(): p.requires_grad = True
def set_mode(m):
    m.eval()      # BatchNorm nech zostane na štatistikách ImageNetu
    head.train()
```
Trénuje sa len hlava. **Prečo `eval()` na zmrazenom backbone:** aj so zmrazenými váhami by BatchNorm v režime `train()` aktualizoval svoje kĺzavé štatistiky z batchov jedla a posúval by sa. ConvNeXt probe: 91,87 % na validácii za 6 minút.

**Fine-tuning** (`param_groups`):
```python
[{"params": body, "lr": lr}, {"params": head, "lr": lr * 10}]
```
Trénuje sa všetko, ale náhodná hlava sa učí 10× rýchlejšie než predtrénované telo, ktoré treba len postrčiť. Príliš vysoký rate na tele by zničil, čo sa naučilo na ImageNete.

**Finál:** ConvNeXt-Tiny s fine-tuningom, lr 1e-4, silná augmentácia, 3 seedy: **94,76 % ± 0,68** na teste. Ponaučenie: **výber predtrénovaného backbonu bol dôležitejší než spôsob, ako ho trénuješ**; zmrazené príznaky ConvNeXt porazili ostatné dva aj s fine-tuningom.

---

## Krok 7 · Analýza chýb cez maticu zámen (`confusion`, `top_confusions`)

```python
m = np.zeros((n, n), int)
np.add.at(m, (y, probs.argmax(1)), 1)     # riadok = skutočná trieda, stĺpec = predpovedaná
```

`np.add.at` správne spracuje opakované dvojice (skutočná, predpovedaná) (obyčajné `m[y, pred] += 1` by každú dvojicu započítalo len raz). Maximum mimo diagonály ukáže, ktoré triedy sa zamieňajú: mačka ↔ pes je 176 z 861 chýb na CIFAR; jablkový koláč ↔ chlebový puding 44 zo 115 chýb na Food-10. **Jedno číslo presnosti skryje, že jedna dvojica spôsobuje 38 % chýb.**

---

## Krok 8 · Grad-CAM: kam sa model pozeral? ([`gradcam.py`](gradcam.py))

```python
handle = layer.register_forward_hook(lambda m, i, out: store.update(a=out))
score = model(x)[0, cls]
grads = torch.autograd.grad(score, store["a"])[0]
cam = F.relu((grads.mean((2, 3), keepdim=True) * store["a"]).sum(1, keepdim=True))
cam = F.interpolate(cam, x.shape[2:], mode="bilinear")
```

**Ako to funguje:**
1. **Forward hook** si uloží posledné konvolučné mapy príznakov `A` bez zmeny modelu.
2. `torch.autograd.grad` dá `∂skóre/∂A`: o koľko by každá bunka každého kanála zmenila skóre triedy.
3. **Spriemeruj gradient cez mapu** → jedna váha na kanál („aký dôležitý je tento kanál pre túto triedu“).
4. **Vážený súčet kanálov** → jedna heatmapa; **ReLU** nechá len oblasti, ktoré hovoria *za* triedu.
5. Zväčši na veľkosť obrázka.

**Prečo je tu priemerovanie správne:** tieto klasifikátory končia global average poolingom, čo je presne prípad, pre ktorý bol Grad-CAM odvodený. (Kapitola 11 v Racing Lab ukazuje, čo sa pokazí, keď hlava namiesto toho splošťuje.)

**Čo ukázal:** na CIFAR mapy pokrývajú objekt aj pri chybách. Na Food-10 padá do stredovej štvrtiny len 22,6 % heatmapy (rovnomerná by mala 25 %): jedlo sa rozpoznáva podľa textúry rozloženej po celom zábere. Tieto mapy ber ako náznak, nie dôkaz.

---

## Krok 9 · Export do ONNX ([`export_onnx.py`](export_onnx.py))

```python
torch.onnx.export(model, (dummy,), "food10.onnx",
                  input_names=["image"], output_names=["logits"],
                  dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=18)
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
```

**Prečo ONNX:** prenosný súbor, ktorý beží bez PyTorchu (ONNX Runtime, mobil, iné jazyky). `dynamic_axes` povolí ľubovoľnú veľkosť batchu.

**Ako sa to overí:** celá testovacia sada prejde PyTorchom aj ONNX Runtime: rovnaká presnosť (95,40 %), max rozdiel logitov 2,8e-5. **Latencia** (medián zo 100, meraná *pred* spustením data loadera, aby jeho procesy nesúperili o jadrá CPU): 299 ms PyTorch vs 15 ms ONNX Runtime na CPU.

---

## Krok 10 · Spusti všetko, bezpečne

```bash
nohup caffeinate -i sh run_all.sh > train.log 2>&1 &   # ~5 hodín na M4 Pro
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Prečo odpojene a s ukladaním po každom behu:** raz tréning zomrel spolu s reláciou terminálu a stratila sa hodina práce. Teraz každá fáza uloží výsledky hneď, ako skončí, a `caffeinate` drží Mac hore.

---

## Vyskúšaj sám

1. Rozšír mriežku learning rate na CIFAR nad 0,3. Najlepší výsledok bol na vrchu mriežky; je optimum vyššie?
2. Sprav fine-tuning ConvNeXt s rovnakým learning rate pre telo aj hlavu (odstráň ×10). Koľko presnosti stratíš?
3. Odstráň `m.eval()` vo `freeze_backbone` a spusti linear probe znova. Koľko stojí posun BatchNorm?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 2 | Validačná časť na voľby, test len pre finálne modely; augmentuj len tréning |
| 3 | Päťriadkový cyklus; režimy train/eval kvôli BatchNorm; one-cycle LR; najlepšie váhy podľa validácie |
| 4 | Konvolúcia = zdieľané lokálne filtre; reziduálny blok = uč sa korekciu F(x) + x |
| 5 | Najprv lad learning rate; augmentácia je druhá |
| 6 | Predtrénované príznaky porazia tréning od nuly na malých dátach; hlava sa učí rýchlejšie než telo |
| 7 | Matica zámen ukáže, ktorá dvojica tried spôsobuje chyby |
| 8 | Grad-CAM = mapy príznakov vážené gradientom; náznak, nie dôkaz |
| 9 | ONNX: rovnaká presnosť, prenosný, rýchlejší na CPU |
