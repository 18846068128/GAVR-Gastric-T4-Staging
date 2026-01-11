import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from data.io import load_feature_tables, load_labels, get_split_ids, extract_by_ids
from data.dataset import MultimodalFeatureDataset
from models.vit_fusion import ViTFusion
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.plots import save_training_curves, save_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser("ViT-based multimodal fusion training")
    p.add_argument("--label_csv", type=str, required=True)
    p.add_argument("--radiomics_csv", type=str, required=True)
    p.add_argument("--feat2d_csv", type=str, required=True)
    p.add_argument("--feat3d_csv", type=str, required=True)
    p.add_argument("--clinical_csv", type=str, required=True)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.1)

    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_classes", type=int, default=2)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="outputs")
    return p.parse_args()


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for (r, f2d, f3d, c), labels in loader:
        r = r.to(device); f2d = f2d.to(device); f3d = f3d.to(device); c = c.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(r, f2d, f3d, c)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        pred = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc, _ = compute_metrics(y_true, y_pred)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    for (r, f2d, f3d, c), labels in loader:
        r = r.to(device); f2d = f2d.to(device); f3d = f3d.to(device); c = c.to(device)
        labels = labels.to(device)

        logits = model(r, f2d, f3d, c)
        loss = criterion(logits, labels)

        total_loss += float(loss.item())
        pred = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc, cm = compute_metrics(y_true, y_pred)
    return avg_loss, acc, cm


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load
    label_df = load_labels(args.label_csv)
    r_df, f2d_df, f3d_df, c_df = load_feature_tables(
        args.radiomics_csv, args.feat2d_csv, args.feat3d_csv, args.clinical_csv
    )
    train_ids, test_ids, y_train, y_test = get_split_ids(label_df)

    # extract arrays by ids
    r_train = extract_by_ids(r_df, train_ids)
    f2d_train = extract_by_ids(f2d_df, train_ids)
    f3d_train = extract_by_ids(f3d_df, train_ids)
    c_train = extract_by_ids(c_df, train_ids)

    r_test = extract_by_ids(r_df, test_ids)
    f2d_test = extract_by_ids(f2d_df, test_ids)
    f3d_test = extract_by_ids(f3d_df, test_ids)
    c_test = extract_by_ids(c_df, test_ids)

    # standardize per modality (fit on train only)
    scalers = [StandardScaler() for _ in range(4)]
    r_train = scalers[0].fit_transform(r_train); r_test = scalers[0].transform(r_test)
    f2d_train = scalers[1].fit_transform(f2d_train); f2d_test = scalers[1].transform(f2d_test)
    f3d_train = scalers[2].fit_transform(f3d_train); f3d_test = scalers[2].transform(f3d_test)
    c_train = scalers[3].fit_transform(c_train); c_test = scalers[3].transform(c_test)

    # dataset/loader
    train_ds = MultimodalFeatureDataset(r_train, f2d_train, f3d_train, c_train, y_train)
    test_ds = MultimodalFeatureDataset(r_test, f2d_test, f3d_test, c_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    input_dims = (r_train.shape[1], f2d_train.shape[1], f3d_train.shape[1], c_train.shape[1])

    model = ViTFusion(
        input_dims=input_dims,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        dropout=args.dropout,
        num_classes=args.num_classes,
        use_modality_type_embed=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, va_cm = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(va_loss)
        train_accs.append(tr_acc); val_accs.append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            save_confusion_matrix(va_cm, os.path.join(args.save_dir, "confusion_matrix_best.png"))

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    save_training_curves(train_losses, val_losses, train_accs, val_accs,
                         os.path.join(args.save_dir, "training_curves.png"))

    print(f"✅ best val acc = {best_acc:.4f}, saved to {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
