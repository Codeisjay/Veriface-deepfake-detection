import os, argparse, json, random, numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm
from model import build_model

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(train_dir, val_dir, img_size, batch_size, num_workers):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes

def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(y.numpy().tolist())
    auc = roc_auc_score(y_true, y_prob)
    preds = (np.array(y_prob) >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    return {"auc": float(auc), "acc": float(acc), "f1": float(f1)}

def main(cfg_path):
    cfg = json.load(open(cfg_path))
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = get_loaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        cfg["data"]["img_size"], cfg["train"]["batch_size"],
        cfg["data"]["num_workers"]
    )
    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"].get("mixed_precision", True))

    best_metric = -1.0
    out_dir = Path(cfg["output"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['train']['epochs']}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=cfg["train"].get("mixed_precision", True)):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=float(loss))

        metrics = evaluate(model, val_loader, device)
        print(f"val: {metrics}")
        key = cfg["output"]["save_best_by"]
        score = metrics["auc"] if key == "val_auc" else metrics.get("acc", 0.0)

        if score > best_metric:
            best_metric = score
            ckpt = out_dir / "best.ckpt"
            torch.save({"model": model.state_dict(), "classes": classes, "cfg": cfg}, ckpt)
            print(f"[saved] {ckpt} (metric={best_metric:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.json")
    args = ap.parse_args()
    main(args.config)
