import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from loader import load_decision_tree, load_knn, load_naive_bayes

app = Flask(__name__)

IMG_SIZE   = (64, 64)
MODELS_DIR = "models"

# ─── Load all models once at startup ──────────────────────
print("Loading models...")
dt_model,  dt_classes,  dt_metrics,  dt_cm          = load_decision_tree(f"{MODELS_DIR}/decision_tree.h5")
knn_model, knn_pca, knn_classes, knn_metrics, knn_cm = load_knn(f"{MODELS_DIR}/knn.h5")
nb_model,  nb_classes,  nb_metrics,  nb_cm           = load_naive_bayes(f"{MODELS_DIR}/naive_bayes.h5")
print("Models loaded.")

MODELS = {
    "Decision Tree": {
        "model":   dt_model,
        "pca":     None,
        "classes": dt_classes,
        "metrics": dt_metrics,
        "cm":      dt_cm,
    },
    "KNN": {
        "model":   knn_model,
        "pca":     knn_pca,        # apply PCA before predict
        "classes": knn_classes,
        "metrics": knn_metrics,
        "cm":      knn_cm,
    },
    "Naive Bayes": {
        "model":   nb_model,
        "pca":     None,
        "classes": nb_classes,
        "metrics": nb_metrics,
        "cm":      nb_cm,
    },
}


def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).flatten().astype(np.float32) / 255.0
    return arr.reshape(1, -1)


@app.route("/")
def index():
    # Pass model metrics and confusion matrices to template
    model_stats = {}
    for name, data in MODELS.items():
        model_stats[name] = {
            "metrics": data["metrics"],
            "cm":      data["cm"],
            "classes": data["classes"],
        }
    return render_template("index.html", model_stats=model_stats)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file  = request.files["image"]
    img_bytes = file.read()
    X = preprocess(img_bytes)

    results = {}
    for name, data in MODELS.items():
        clf     = data["model"]
        pca     = data["pca"]
        classes = data["classes"]
        X_input = pca.transform(X) if pca is not None else X
        pred    = int(clf.predict(X_input)[0])
        proba   = clf.predict_proba(X_input)[0].tolist()
        results[name] = {
            "prediction":    classes[pred],
            "confidence":    round(max(proba) * 100, 2),
            "probabilities": {classes[i]: round(p * 100, 2) for i, p in enumerate(proba)},
        }

    # Also return the preview image as base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    ext     = file.content_type.split("/")[-1]

    return jsonify({
        "results":  results,
        "image_b64": f"data:{file.content_type};base64,{img_b64}",
    })


if __name__ == "__main__":
    app.run(debug=True)
