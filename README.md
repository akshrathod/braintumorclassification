# BrainTumorClassifier (Research/Education)

**DenseNet201** transfer learning for **brain tumor classification** from MRI images. Includes:
- Training pipeline (PyTorch + torchvision)
- Evaluation with confusion matrix & classification report
- Flask inference API (file upload) + simple HTML
- Docker for serving the model
- ⚠️ **Disclaimer:** This project is for research/education only; **not for clinical use**.

## Quickstart

### 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare data
Organize images as ImageFolder layout:
```
data/
  train/
    glioma/ *.png|jpg
    meningioma/ ...
    pituitary/ ...
    notumor/ ...
  val/
    glioma/ ...
    meningioma/ ...
    pituitary/ ...
    notumor/ ...
```
> You can use public datasets like Kaggle's 4-class MRI (7023 images) or Br35H (2-class). See **DATASETS.md**.

### 3) Train
```bash
python app/train.py --train data/train --val data/val --epochs 10 --batch 16 --lr 3e-4 --img 224
# model saved to artifacts/densenet201_best.pt
```

### 4) Evaluate
```bash
python app/eval.py --val data/val --weights artifacts/densenet201_best.pt
```

### 5) Serve (Flask)
```bash
FLASK_ENV=production FLASK_APP=service/app.py python -m flask run -p 5000
# open http://127.0.0.1:5000 and upload an image
```

### 6) Docker (serving only)
```bash
docker build -t brain-tumor-flask:latest .
docker run --rm -p 5000:5000 -v $(pwd)/artifacts:/app/artifacts brain-tumor-flask:latest
```

## Project Structure
```
BrainTumorClassifier/
  app/
    train.py          # training w/ DenseNet201
    eval.py           # metrics & confusion matrix
    utils.py          # dataset & transforms
    gradcam.py        # optional Grad-CAM visualization
  service/
    app.py            # Flask app (upload + /predict API)
    templates/
      index.html
  data/               # put your train/val folders here
  artifacts/          # saved weights
  tests/
    test_transforms.py
  requirements.txt
  Dockerfile
  DATASETS.md
  README.md
  LICENSE
```

## Notes
- Default classes: `['glioma','meningioma','pituitary','notumor']` — change via `--classes`.
- Mixed precision (autocast) enabled if CUDA available.
- Strong augmentations (flip/rotate/elastic) are optional flags.



---

## Using the Kaggle Brain Tumor MRI Dataset (4 classes / 7,023 images)

**Dataset**: Brain Tumor MRI (classes: `glioma`, `meningioma`, `pituitary`, `notumor`).

### Download via Kaggle CLI
1) Install & auth:
```bash
pip install kaggle
# place your Kaggle API token at ~/.kaggle/kaggle.json
```
2) Download (either mirror works; first is most used):
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d raw_kaggle
```
Typical layout inside `raw_kaggle`:
```
Brain Tumor MRI Dataset/
  Testing/
    glioma_tumor/
    meningioma_tumor/
    pituitary_tumor/
    no_tumor/
  Training/
    glioma_tumor/
    meningioma_tumor/
    pituitary_tumor/
    no_tumor/
```

3) Create our standard train/val split:
```bash
python split_data.py \
  --src "raw_kaggle/Brain Tumor MRI Dataset" \
  --train_sub Training --val_sub Testing \
  --dest data \
  --map_names  # converts folder names to (glioma|meningioma|pituitary|notumor)
```
This produces:
```
data/
  train/{glioma,meningioma,pituitary,notumor}/...
  val/{glioma,meningioma,pituitary,notumor}/...
```


## CI & One‑Click Deploy

**GitHub Actions (CI):** enabled via `.github/workflows/ci.yml` (lint + tests).  
Add this badge to your README after pushing to GitHub (replace `YOUR-USER`):

```
[![CI](https://github.com/YOUR-USER/BrainTumorClassifier/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USER/BrainTumorClassifier/actions/workflows/ci.yml)
```

**Heroku (one click):** after you push to GitHub, replace `YOUR-USER` and click:  
```
[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/YOUR-USER/BrainTumorClassifier)
```

**Render (one click):** after pushing, use `render.yaml` in the repo or open the Render dashboard and **New → Web Service → Deploy from repo**. The project is Docker-ready.

