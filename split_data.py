import argparse, os, shutil, re
from pathlib import Path

CANONICAL = {"glioma","meningioma","pituitary","notumor"}

def normalize(name: str) -> str:
    n = name.lower()
    n = n.replace(" ", "_")
    # map common variants
    n = n.replace("no_tumor", "notumor")
    n = n.replace("no-tumor", "notumor")
    n = n.replace("no.tumor", "notumor")
    n = n.replace("pituitary_tumor", "pituitary")
    n = n.replace("meningioma_tumor", "meningioma")
    n = n.replace("glioma_tumor", "glioma")
    # strip common suffix/prefix noise
    n = re.sub(r"(?i)_?tumou?r$", "", n)
    n = re.sub(r"[^a-z0-9_]", "", n)
    return n


    

def copy_tree(src: Path, dst: Path, map_names: bool):
    dst.mkdir(parents=True, exist_ok=True)
    for cls_dir in sorted([p for p in src.iterdir() if p.is_dir()]):
        cls_name = normalize(cls_dir.name) if map_names else cls_dir.name
        out_cls = dst / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)
        for img in cls_dir.rglob("*"):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                shutil.copy2(img, out_cls / img.name)

def main():
    ap = argparse.ArgumentParser(description="Split Kaggle Brain Tumor dataset to data/train & data/val")
    ap.add_argument("--src", required=True, help="Root that contains Training/ and Testing/ (from Kaggle)")
    ap.add_argument("--train_sub", default="Training", help="Subfolder for training data")
    ap.add_argument("--val_sub", default="Testing", help="Subfolder for validation data")
    ap.add_argument("--dest", default="data", help="Destination root (will create train/ and val/)")
    ap.add_argument("--map_names", action="store_true", help="Normalize class folder names to canonical set")
    args = ap.parse_args()

    src = Path(args.src)
    train_src = src / args.train_sub
    val_src = src / args.val_sub
    dest = Path(args.dest)
    train_dst = dest / "train"
    val_dst = dest / "val"

    if not train_src.exists() or not val_src.exists():
        raise SystemExit(f"Could not find {train_src} and/or {val_src}")

    print(f"Copying training from {train_src} -> {train_dst}")
    copy_tree(train_src, train_dst, args.map_names)
    print(f"Copying validation from {val_src} -> {val_dst}")
    copy_tree(val_src, val_dst, args.map_names)

    # Verify classes
    got = sorted([p.name for p in train_dst.iterdir() if p.is_dir()])
    print("Train classes:", got)
    if args.map_names:
        missing = CANONICAL.difference(set(got))
        if missing:
            print("Warning: missing canonical classes:", sorted(missing))

if __name__ == "__main__":
    main()
