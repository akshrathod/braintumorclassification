from flask import Flask, request, render_template, jsonify
import torch, io
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

app = Flask(__name__, template_folder="templates")

# Load model at import so both `python app.py` and `flask run` work
load_model()

CLASSES = None
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weights_path="artifacts/densenet201_best.pt"):
    global MODEL, CLASSES
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    in_feats = model.classifier.in_features
    model.classifier = nn.Linear(in_feats, 4)  # default 4 classes; overriden by weights
    ckpt = torch.load(weights_path, map_location=DEVICE)
    CLASSES = ckpt.get("classes", ["glioma","meningioma","pituitary","notumor"])
    model.classifier = nn.Linear(in_feats, len(CLASSES))
    model.load_state_dict(ckpt["model"])
    MODEL = model.to(DEVICE).eval()

def preprocess(img, img_size=224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return tfm(img).unsqueeze(0)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", classes=CLASSES)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    img = Image.open(io.BytesIO(f.read())).convert("RGB")
    x = preprocess(img).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    result = [{"class": c, "prob": float(p)} for c, p in zip(CLASSES, probs)]
    return jsonify({"predictions": sorted(result, key=lambda r: r["prob"], reverse=True)})

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
