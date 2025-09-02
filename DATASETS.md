# Datasets

## Kaggle: Brain Tumor MRI (4 classes; ~7,023 images)

- **Classes**: glioma, meningioma, pituitary, notumor
- **Typical source structure**:
```
Brain Tumor MRI Dataset/
  Training/{glioma_tumor,meningioma_tumor,pituitary_tumor,no_tumor}
  Testing/{glioma_tumor,meningioma_tumor,pituitary_tumor,no_tumor}
```

### Steps

1. Install Kaggle CLI and authenticate:
```bash
pip install kaggle
# put API token at ~/.kaggle/kaggle.json
```

2. Download and unzip:
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d raw_kaggle
```

3. Split and normalize class names:
```bash
python split_data.py --src "raw_kaggle/Brain Tumor MRI Dataset" --train_sub Training --val_sub Testing --dest data --map_names
```

> After this, you can run training:
```bash
python app/train.py --train data/train --val data/val --epochs 10 --batch 16 --img 224
```
