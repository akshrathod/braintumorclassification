import os, argparse, time, json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from sklearn.metrics import accuracy_score
from app.utils import make_dataloaders

def build_model(num_classes: int):
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    return model

def train_epoch(model, loader, device, criterion, optimizer, scaler):
    model.train()
    losses = []
    all_pred, all_true = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=device.type=="cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        all_pred.extend(logits.argmax(1).detach().cpu().tolist())
        all_true.extend(labels.detach().cpu().tolist())
    acc = accuracy_score(all_true, all_pred)
    return sum(losses)/len(losses), acc

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    losses = []
    all_pred, all_true = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        all_pred.extend(logits.argmax(1).detach().cpu().tolist())
        all_true.extend(labels.detach().cpu().tolist())
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_true, all_pred)
    return sum(losses)/len(losses), acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train folder (ImageFolder)")
    ap.add_argument("--val", required=True, help="val folder (ImageFolder)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--classes", nargs="*", default=['glioma','meningioma','pituitary','notumor'])
    ap.add_argument("--out", default="artifacts/densenet201_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_dataloaders(args.train, args.val, args.img, args.batch, aug=args.aug)
    print("Detected classes:", classes)

    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=device.type=="cuda")

    best_acc, best_path = 0.0, args.out
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc = eval_epoch(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes}, best_path)
            print(f"Saved best model to {best_path} (val_acc={best_acc:.4f})")

    with open("artifacts/metrics.json", "w") as f:
        json.dump({"best_val_acc": best_acc}, f, indent=2)

if __name__ == "__main__":
    main()
