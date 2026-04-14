import numpy as np
import h5py
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree._tree import Tree, NODE_DTYPE


# ─── DECISION TREE ────────────────────────────────────────
def load_decision_tree(h5_path):
    with h5py.File(h5_path, "r") as f:
        n_classes  = int(f.attrs["n_classes"])
        n_features = int(f.attrs["n_features"])
        metrics    = _read_metrics(f)
        classes    = [c.decode() for c in f["classes"][:]]
        cm         = f["confusion_matrix"][:]

        children_left  = f["children_left"][:]
        children_right = f["children_right"][:]
        feature        = f["feature"][:]
        threshold      = f["threshold"][:]
        value          = f["value"][:]
        n_node_samples = f["n_node_samples"][:]
        impurity       = f["impurity"][:]

    n_nodes  = children_left.shape[0]
    nodes    = np.zeros(n_nodes, dtype=NODE_DTYPE)
    nodes["left_child"]               = children_left
    nodes["right_child"]              = children_right
    nodes["feature"]                  = feature
    nodes["threshold"]                = threshold
    nodes["impurity"]                 = impurity
    nodes["n_node_samples"]           = n_node_samples
    nodes["weighted_n_node_samples"]  = n_node_samples.astype(np.float64)

    tree_obj = Tree(n_features, np.array([n_classes], dtype=np.intp), 1)
    tree_obj.__setstate__({
        "max_depth":  int(np.max(np.where(children_left == -1, 0, np.arange(n_nodes)))),
        "node_count": n_nodes,
        "nodes":      nodes,
        "values":     value,
    })

    clf = DecisionTreeClassifier()
    clf.tree_          = tree_obj
    clf.n_classes_     = n_classes
    clf.n_features_in_ = n_features
    clf.classes_       = np.arange(n_classes)
    clf.n_outputs_     = 1

    return clf, classes, metrics, cm.tolist()


# ─── KNN ──────────────────────────────────────────────────
def load_knn(h5_path):
    with h5py.File(h5_path, "r") as f:
        n_neighbors    = int(f.attrs["n_neighbors"])
        metric         = str(f.attrs["metric"])
        algorithm      = str(f.attrs["algorithm"])
        pca_components = int(f.attrs["pca_components"])
        metrics        = _read_metrics(f)
        classes        = [c.decode() for c in f["classes"][:]]
        cm             = f["confusion_matrix"][:]
        fit_X          = f["fit_X"][:]
        y              = f["y"][:]
        components_    = f["pca_components"][:]
        mean_          = f["pca_mean"][:]
        explained_var_ = f["pca_explained_var"][:]

    # Rebuild PCA without refitting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_components)
    pca.components_         = components_
    pca.mean_               = mean_
    pca.explained_variance_ = explained_var_
    pca.n_components_       = pca_components
    pca.n_features_in_      = mean_.shape[0]

    # Rebuild KNN
    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm=algorithm
    )
    # Use fit() directly on the PCA-reduced data — fast since dimensions are small
    clf.fit(fit_X, y)

    return clf, pca, classes, metrics, cm.tolist()


# ─── NAIVE BAYES ──────────────────────────────────────────
def load_naive_bayes(h5_path):
    with h5py.File(h5_path, "r") as f:
        metrics     = _read_metrics(f)
        classes     = [c.decode() for c in f["classes"][:]]
        cm          = f["confusion_matrix"][:]
        theta       = f["theta"][:]
        var         = f["var"][:]
        class_prior = f["class_prior"][:]
        nb_classes  = f["nb_classes"][:]

    clf = GaussianNB()
    clf.theta_         = theta
    clf.var_           = var
    clf.class_prior_   = class_prior
    clf.classes_       = nb_classes
    clf.n_features_in_ = theta.shape[1]

    return clf, classes, metrics, cm.tolist()


# ─── HELPER ───────────────────────────────────────────────
def _read_metrics(f):
    return {
        "accuracy":  round(float(f.attrs["accuracy"]),  4),
        "precision": round(float(f.attrs["precision"]), 4),
        "recall":    round(float(f.attrs["recall"]),    4),
        "f1":        round(float(f.attrs["f1"]),        4),
    }
