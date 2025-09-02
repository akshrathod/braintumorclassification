import argparse, torch, json, os
import torch.nn as nn
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from app.utils import make_dataloaders

def load_model(weights_path, num_classes):
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, num_classes)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    classes = ckpt.get("classes", None)
    return model, classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()

    _, val_loader, classes = make_dataloaders("data/train", args.val, args.img, batch_size=16, aug=False)
    model, saved_classes = load_model(args.weights, num_classes=len(classes))
    if saved_classes is not None:
        classes = saved_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
