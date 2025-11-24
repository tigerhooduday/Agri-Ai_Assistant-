# src/cv_module/train.py  (REPLACE your current file with this)
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

from src.cv_module.dataset import PlantVillageDataset

# helper to pick backbone
def build_model(num_classes, backbone="resnet50", pretrained=True):
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone: " + backbone)
    return model

def freeze_backbone_params(model, backbone="resnet50"):
    # freeze all parameters except final classification layer(s)
    backbone = backbone.lower()
    if backbone.startswith("resnet"):
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = False
    elif backbone == "mobilenet_v2":
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False
    elif backbone.startswith("efficientnet"):
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

def train_epoch(model, loader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss / total, correct / total

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all = []
    labels_all = []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds_all.append(outputs.softmax(dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    preds_all = np.vstack(preds_all)
    labels_all = np.concatenate(labels_all)
    return running_loss / len(labels_all), preds_all, labels_all

def make_label_map(df_csv):
    import pandas as pd
    df = pd.read_csv(df_csv)
    labels = sorted(df['label'].unique())
    label2idx = {l: i for i, l in enumerate(labels)}
    idx2label = {i: l for l, i in label2idx.items()}
    return label2idx, idx2label

def main(args):
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = "cuda" if use_cuda else "cpu"
    print("Using device:", device)

    # label maps
    label2idx, idx2label = make_label_map(args.meta)
    num_classes = len(label2idx)

    # save label map
    models_dir = Path(args.output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "labels.json", "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

    # transforms (pass to dataset)
    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_csv = args.train_csv or "data/processed/train.csv"
    val_csv = args.val_csv or "data/processed/val.csv"
    train_ds = PlantVillageDataset(csv_file=train_csv, transforms=train_tf)
    val_ds = PlantVillageDataset(csv_file=val_csv, transforms=val_tf)

    # ensure dataset label mapping exists
    if not hasattr(train_ds, "label2idx"):
        train_ds.label2idx = label2idx
        val_ds.label2idx = label2idx

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, args.num_workers//2), pin_memory=use_cuda)

    # adjust default backbone for CPU
    if device == "cpu" and args.backbone == "resnet50" and not args.no_pretrained:
        print("CPU detected: switching default backbone to resnet18 for speed.")
        args.backbone = "resnet18"

    model = build_model(num_classes, backbone=args.backbone, pretrained=not args.no_pretrained)
    if args.freeze_backbone:
        freeze_backbone_params(model, backbone=args.backbone)
        print("Backbone frozen; only classifier will be trained.")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # only pass parameters that require grad to optimizer (so frozen params excluded)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    tb = SummaryWriter(log_dir=str(models_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")))

    use_amp = use_cuda and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp, scaler=scaler)
        val_loss, val_preds, val_labels = eval_model(model, val_loader, criterion, device)
        val_pred_idx = val_preds.argmax(axis=1)
        val_acc = (val_pred_idx == val_labels).mean()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        tb.add_scalar("train/loss", train_loss, epoch)
        tb.add_scalar("train/acc", train_acc, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)
        tb.add_scalar("val/acc", val_acc, epoch)

        scheduler.step(val_loss)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "label2idx": label2idx,
        }
        torch.save(ckpt, models_dir / f"ckpt_epoch_{epoch}.pth")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, models_dir / "best.pth")
            print("Saved best checkpoint.")

    print("Best val acc:", best_val_acc)
    pred_idx = val_preds.argmax(axis=1)
    report = classification_report(val_labels, pred_idx, target_names=[idx2label[i] for i in range(num_classes)], zero_division=0)
    print(report)
    with open(models_dir / "val_classification_report.txt", "w") as f:
        f.write(report)
    tb.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="data/processed/metadata.csv")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--output-dir", default="models/disease_model")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet18","resnet50","mobilenet_v2","efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    main(args)
