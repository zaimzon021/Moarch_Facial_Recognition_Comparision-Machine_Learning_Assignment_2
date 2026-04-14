import os
import numpy as np
import h5py
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ─── CONFIG ───────────────────────────────────────────────
IMG_SIZE   = (64, 64)
DATA_DIR   = "Data/Training"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── 1. LOAD IMAGES ───────────────────────────────────────
def load_dataset(data_dir, img_size):
    X, y = [], []
    classes = sorted(os.listdir(data_dir))
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    print(f"Classes: {label_map}")

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = os.listdir(cls_path)
        print(f"  Loading {len(files)} images from '{cls}'...")
        for fname in files:
            fpath = os.path.join(cls_path, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(img_size)
                X.append(np.array(img).flatten().astype(np.float32) / 255.0)
                y.append(label_map[cls])
            except Exception as e:
                print(f"    Skipping {fname}: {e}")

    return np.array(X), np.array(y), classes

X, y, classes = load_dataset(DATA_DIR, IMG_SIZE)
print(f"\nDataset: X={X.shape}, y={y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 2. METRICS HELPER ────────────────────────────────────
def compute_metrics(y_true, y_pred, n_classes):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return acc, prec, rec, f1, cm

# ─── 3. TRAIN ─────────────────────────────────────────────
print("\n--- Training Decision Tree ---")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc, dt_prec, dt_rec, dt_f1, dt_cm = compute_metrics(y_test, dt_pred, len(classes))
print(f"  Acc={dt_acc:.4f}  Prec={dt_prec:.4f}  Rec={dt_rec:.4f}  F1={dt_f1:.4f}")

print("\n--- Training KNN (with PCA: 12288 → 100 dims) ---")
PCA_COMPONENTS = 100
pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", algorithm="ball_tree")
knn.fit(X_train_pca, y_train)
knn_pred = knn.predict(X_test_pca)
knn_acc, knn_prec, knn_rec, knn_f1, knn_cm = compute_metrics(y_test, knn_pred, len(classes))
print(f"  Acc={knn_acc:.4f}  Prec={knn_prec:.4f}  Rec={knn_rec:.4f}  F1={knn_f1:.4f}")

print("\n--- Training Naive Bayes ---")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_acc, nb_prec, nb_rec, nb_f1, nb_cm = compute_metrics(y_test, nb_pred, len(classes))
print(f"  Acc={nb_acc:.4f}  Prec={nb_prec:.4f}  Rec={nb_rec:.4f}  F1={nb_f1:.4f}")

# ─── 4. SAVE DECISION TREE ────────────────────────────────
print("\n--- Saving Decision Tree ---")
t = dt.tree_
with h5py.File(os.path.join(MODELS_DIR, "decision_tree.h5"), "w") as f:
    f.attrs["model_type"] = "DecisionTree"
    f.attrs["n_classes"]  = int(dt.n_classes_)
    f.attrs["n_features"] = int(dt.n_features_in_)
    f.attrs["max_depth"]  = int(dt.max_depth) if dt.max_depth else -1
    f.attrs["accuracy"]   = float(dt_acc)
    f.attrs["precision"]  = float(dt_prec)
    f.attrs["recall"]     = float(dt_rec)
    f.attrs["f1"]         = float(dt_f1)

    f.create_dataset("classes",         data=np.array(classes, dtype="S10"))
    f.create_dataset("confusion_matrix",data=dt_cm)
    f.create_dataset("children_left",   data=t.children_left)
    f.create_dataset("children_right",  data=t.children_right)
    f.create_dataset("feature",         data=t.feature)
    f.create_dataset("threshold",       data=t.threshold)
    f.create_dataset("value",           data=t.value)
    f.create_dataset("n_node_samples",  data=t.n_node_samples)
    f.create_dataset("impurity",        data=t.impurity)
print("  Saved: models/decision_tree.h5")

# ─── 5. SAVE KNN ──────────────────────────────────────────
print("\n--- Saving KNN ---")
with h5py.File(os.path.join(MODELS_DIR, "knn.h5"), "w") as f:
    f.attrs["model_type"]    = "KNN"
    f.attrs["n_neighbors"]   = int(knn.n_neighbors)
    f.attrs["metric"]        = knn.metric
    f.attrs["algorithm"]     = knn.algorithm
    f.attrs["pca_components"]= int(PCA_COMPONENTS)
    f.attrs["accuracy"]      = float(knn_acc)
    f.attrs["precision"]     = float(knn_prec)
    f.attrs["recall"]        = float(knn_rec)
    f.attrs["f1"]            = float(knn_f1)

    f.create_dataset("classes",           data=np.array(classes, dtype="S10"))
    f.create_dataset("confusion_matrix",  data=knn_cm)
    f.create_dataset("fit_X",             data=knn._fit_X)   # PCA-reduced training data
    f.create_dataset("y",                 data=knn._y)
    # Save PCA internals so we can transform new images at prediction time
    f.create_dataset("pca_components",    data=pca.components_)
    f.create_dataset("pca_mean",          data=pca.mean_)
    f.create_dataset("pca_explained_var", data=pca.explained_variance_)
print("  Saved: models/knn.h5")

# ─── 6. SAVE NAIVE BAYES ──────────────────────────────────
print("\n--- Saving Naive Bayes ---")
with h5py.File(os.path.join(MODELS_DIR, "naive_bayes.h5"), "w") as f:
    f.attrs["model_type"] = "GaussianNB"
    f.attrs["accuracy"]   = float(nb_acc)
    f.attrs["precision"]  = float(nb_prec)
    f.attrs["recall"]     = float(nb_rec)
    f.attrs["f1"]         = float(nb_f1)

    f.create_dataset("classes",          data=np.array(classes, dtype="S10"))
    f.create_dataset("confusion_matrix", data=nb_cm)
    f.create_dataset("theta",            data=nb.theta_)
    f.create_dataset("var",              data=nb.var_)
    f.create_dataset("class_prior",      data=nb.class_prior_)
    f.create_dataset("nb_classes",       data=nb.classes_)
print("  Saved: models/naive_bayes.h5")

print("\n✓ Done.")
print(f"  DT   — Acc:{dt_acc:.3f}  P:{dt_prec:.3f}  R:{dt_rec:.3f}  F1:{dt_f1:.3f}")
print(f"  KNN  — Acc:{knn_acc:.3f}  P:{knn_prec:.3f}  R:{knn_rec:.3f}  F1:{knn_f1:.3f}")
print(f"  NB   — Acc:{nb_acc:.3f}  P:{nb_prec:.3f}  R:{nb_rec:.3f}  F1:{nb_f1:.3f}")
