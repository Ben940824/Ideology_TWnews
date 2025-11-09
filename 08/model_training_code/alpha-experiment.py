#!/usr/bin/env python3
"""
Standalone Alpha Experiment for Knowledge Distillation

Usage: python alpha_experiment.py
"""

# --------------------------------------------------
# 0. 安裝 Kaggle 環境需要的套件（在本地執行可自動跳過）
# --------------------------------------------------
import subprocess, sys, os, json, random, math, time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

def install_packages():
    pkgs = [
        "torch", "torchvision", "torchaudio", "torchmetrics",
        "transformers", "datasets", "scikit-learn", "pandas",
        "numpy", "matplotlib", "seaborn", "tqdm"
    ]
    for p in pkgs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
            print(f"Installed {p}")
        except subprocess.CalledProcessError:
            print(f"Failed to install {p}")

try:
    # Kaggle docker image會有環境變數
    if os.environ.get("KAGGLE_URL_BASE"):
        print("Running in Kaggle – installing packages ...")
        install_packages()
except Exception:
    pass  # 本地環境無需安裝

# --------------------------------------------------
# 1. 全域設定
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_metrics(num_classes, device):
    return {
        "acc":          MulticlassAccuracy(num_classes=num_classes, average="micro").to(device),
        "f1_macro":     MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        "f1_weighted":  MulticlassF1Score(num_classes=num_classes, average="weighted").to(device),
        "prec_macro":   MulticlassPrecision(num_classes=num_classes, average="macro").to(device),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro").to(device),
    }

# --------------------------------------------------
# 2. 資料集與 DataLoader
# --------------------------------------------------
class DistillDataset(Dataset):
    """Dataset for knowledge distillation."""
    def __init__(self, df, tokenizer, max_len=512):
        self.df = df
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(
            row["title_content"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze() for k, v in enc.items()}  # still on CPU
        item["label"] = int(row["label_encoded"])
        item["teacher_probs"] = torch.tensor(row["teacher_probs"], dtype=torch.float)
        return item

# --------------------------------------------------
# 3. 評估與訓練函式
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, metrics, return_preds=False, device=None):
    """Evaluate model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    for m in metrics.values():
        m.reset()

    all_preds, all_labels = [], []
    for batch in dataloader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lab  = batch["label"].to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits
        preds  = torch.argmax(logits, dim=1)

        for m in metrics.values():
            m.update(preds, lab)

        if return_preds:
            all_preds.extend(preds.cpu())
            all_labels.extend(lab.cpu())

    results = {n: m.compute().item() for n, m in metrics.items()}

    if return_preds:
        return results, torch.stack(all_preds), torch.stack(all_labels)
    return results

def train(model, dataloader, optimizer, scheduler,
          alpha=0.5, temperature=4.0, class_weights=None):
    """Train one epoch with KD."""
    model.train()
    total_loss = 0
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    ce_loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(device)
    ) if class_weights is not None else nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Training"):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lab  = batch["label"].to(device)
        tprobs = batch["teacher_probs"].to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits

        ce_loss = ce_loss_fn(logits, lab)
        s_logp  = nn.functional.log_softmax(logits / temperature, dim=1)
        t_soft  = nn.functional.softmax(tprobs / temperature, dim=1)
        kl_loss = kl_loss_fn(s_logp, t_soft) * temperature ** 2

        loss = alpha * kl_loss + (1 - alpha) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# --------------------------------------------------
# 4. 輔助：載入 / 產生資料
# --------------------------------------------------
def load_and_prepare_data(path):
    if not os.path.exists(path):
        print(f"Data not found: {path}")
        return None
    df = pd.read_csv(path)
    req = ["title", "content", "title_en", "content_en", "label_encoded"]
    miss = [c for c in req if c not in df.columns]
    if miss:
        print("Missing columns:", miss)
        return None
    df["title_content"] = df["title"] + df["content"]
    df["title_content_en"] = df["title_en"] + df["content_en"]
    return df

def generate_teacher_soft_labels(df, teacher_model_path="launch/POLITICS"):
    print("Generating teacher soft labels ...")
    tok = AutoTokenizer.from_pretrained(teacher_model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        teacher_model_path, num_labels=3
    ).to(device)
    mdl.eval()

    def get_soft(text):
        inp = tok(text, return_tensors="pt", padding=True,
                  truncation=True).to(device)
        with torch.no_grad():
            logit = mdl(**inp).logits
            prob  = torch.nn.functional.softmax(logit, dim=-1)
            prob  = prob[:, [2, 0, 1]]  # reorder if needed
        return prob.squeeze().cpu().numpy()

    tqdm.pandas()
    df["teacher_probs"] = df["title_content_en"].progress_apply(
        lambda x: get_soft(x).tolist()
    )
    return df

# --------------------------------------------------
# 5. Alpha 實驗主流程
# --------------------------------------------------
def run_alpha_experiment(df, alpha_vals, cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    set_seed(42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tr_idx, val_idx = next(iter(kf.split(df)))
    tr_df, val_df = df.iloc[tr_idx], df.iloc[val_idx]

    cls_w = compute_class_weight("balanced",
                                 classes=list(range(cfg["num_classes"])),
                                 y=tr_df["label_encoded"])
    tr_ds = DistillDataset(tr_df, tok, cfg["max_len"])
    val_ds = DistillDataset(val_df, tok, cfg["max_len"])
    tr_dl = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"])

    results = {}
    for a in alpha_vals:
        print(f"\n===== Alpha = {a} =====")
        set_seed(42)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"],
            num_labels=cfg["num_classes"],
            hidden_dropout_prob=cfg["dropout"],
            attention_probs_dropout_prob=cfg["dropout"],
        ).to(device)

        opt = AdamW(mdl.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        steps = len(tr_dl) * cfg["epochs"]
        sched = get_scheduler("linear", optimizer=opt,
                              num_warmup_steps=int(steps*cfg["warmup_ratio"]),
                              num_training_steps=steps)
        metrics = get_metrics(cfg["num_classes"], device)

        best_f1, patience = -1, 0
        history = []
        for ep in range(cfg["epochs"]):
            print(f"Epoch {ep+1}")
            loss = train(mdl, tr_dl, opt, sched, alpha=a,
                         temperature=4.0, class_weights=cls_w)
            ev = evaluate(mdl, val_dl, metrics, device=device)
            print(f"loss={loss:.4f}  eval={ev}")
            history.append({"epoch": ep+1, "train_loss": loss, **ev})

            cur = ev["f1_macro"]
            if cur > best_f1:
                best_f1, patience = cur, 0
                mdl.save_pretrained(f"./alpha_{a}")
                tok.save_pretrained(f"./alpha_{a}")
            else:
                patience += 1
                if patience >= cfg["patience"]:
                    print("Early stop")
                    break

        results[a] = {
            "best_f1_macro": best_f1,
            "final_metrics": ev,
            "epoch_history": history,
        }
    return results

# --------------------------------------------------
# 6. 視覺化與存檔（略）；保留你原本的實作
# --------------------------------------------------
def analyze_results(res, alpha_vals):
    out = []
    for a, r in res.items():
        out.append({"alpha": a, **r["final_metrics"]})
    df = pd.DataFrame(out).sort_values("alpha")
    print("\n===== Summary =====")
    print(df)
    df.to_csv("alpha_experiment_results.csv", index=False)
    best_a = df.iloc[df["f1_macro"].idxmax()]["alpha"]
    print(f"\nBest alpha: {best_a}")
    return df

# --------------------------------------------------
# 7. main
# --------------------------------------------------
def main():
    print("="*60, "\nKnowledge Distillation Alpha Experiment\n", "="*60)

    cfg = dict(
        model_name="ckiplab/bert-base-chinese",
        max_len=512,
        batch_size=16,
        epochs=5,
        lr=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        patience=2,
        num_classes=3,
    )
    alpha_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data_path = "/kaggle/input/taiwan-political-news-dataset/news_training_with_translations.csv"

    df = load_and_prepare_data(data_path)
    if df is None:
        return

    if "teacher_probs" not in df.columns:
        df = generate_teacher_soft_labels(df)

    res = run_alpha_experiment(df, alpha_vals, cfg)
    analyze_results(res, alpha_vals)

if __name__ == "__main__":
    main()