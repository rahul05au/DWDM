"""Classification models with configurable Decision Tree (Gini/Entropy, Max Depth, Pruning),
ID3, J48, Naive Bayes, k-NN, and Random Forest.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report, roc_auc_score
)

TREE_ALGS = ("decision tree", "id3", "j48")


def _prep(num, cat, scale=False):
    """Pipeline for preprocessing numeric and categorical features."""
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))

    transformers = []
    if num:
        transformers.append(('num', Pipeline(steps), num))
    if cat:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat))

    return ColumnTransformer(transformers)


def _feature_names(prep, num, cat):
    """Readable names: numeric -> 'col', one-hot -> 'col = value'."""
    try:
        if 'cat' in prep.named_transformers_:
            cats = prep.named_transformers_['cat'].named_steps['onehot'].categories_
            names = list(num) + [f'{c} = {v}' for c, vals in zip(cat, cats) for v in vals]
            if len(names) == len(prep.get_feature_names_out()):
                return names
    except Exception:
        pass
    try:
        return [n.split('__', 1)[-1] for n in prep.get_feature_names_out()]
    except Exception:
        return list(num) + list(cat)


def run(df: pd.DataFrame, target: str, algorithm: str = 'Decision Tree',
        feature_cols=None, criterion: str = 'gini', max_depth=None,
        min_samples_split: int = 2, min_samples_leaf: int = 1, test_size: float = 0.2):
    """Train and evaluate classification models with complete metrics and diagnostics."""
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # Drop target and select features
    if feature_cols:
        features = [c for c in feature_cols if c != target and c in df.columns]
    else:
        features = [c for c in df.columns if c != target]

    if not features:
        raise ValueError("Need at least one feature column besides the target.")

    # Drop rows where target is NaN
    valid_data = df.dropna(subset=[target]).copy()
    if len(valid_data) < 5:
        raise ValueError("Dataset has too few labeled rows to perform classification (minimum 5).")

    X = valid_data[features]
    y = valid_data[target].astype(str)

    le = LabelEncoder()
    yy = le.fit_transform(y)

    cat = X.select_dtypes(exclude='number').columns.tolist()
    num = X.select_dtypes('number').columns.tolist()

    # Stratified split to preserve class proportions, with graceful fallback
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, yy, test_size=test_size, random_state=42, stratify=yy
        )
    except ValueError:
        # A class too small to stratify
        Xtr, Xte, ytr, yte = train_test_split(
            X, yy, test_size=test_size, random_state=42
        )

    a = algorithm.lower().strip()
    is_tree = a in TREE_ALGS or 'tree' in a

    # Trees do not require scaling so that split thresholds remain in original measurement units
    prep = _prep(num, cat, scale=not is_tree)

    # Configure model
    if a == 'id3':
        # ID3 uses Information Gain / Entropy, without pruning
        est = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif a == 'j48':
        # J48 uses Entropy with post-pruning approximation (ccp_alpha) and min leaf size
        est = DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf=max(2, min_samples_leaf),
            min_samples_split=max(4, min_samples_split),
            ccp_alpha=0.005,
            max_depth=max_depth,
            random_state=42
        )
    elif a in ('decision tree', 'cart'):
        crit = 'entropy' if criterion.lower() in ('entropy', 'information gain') else 'gini'
        est = DecisionTreeClassifier(
            criterion=crit,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif a in ('naïve bayes', 'naive bayes'):
        est = GaussianNB()
    elif a in ('k-nn', 'knn'):
        est = KNeighborsClassifier(n_neighbors=min(5, len(ytr)))
    else:  # Random Forest
        est = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model = Pipeline([('prep', prep), ('model', est)])
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    try:
        proba = model.predict_proba(Xte)
    except Exception:
        proba = None

    feature_names = _feature_names(prep, num, cat)

    # Extract tree diagram and rules if applicable
    tree_text = None
    tree_model_obj = None
    importances = None

    if is_tree:
        tree_model_obj = est
        try:
            tree_text = export_text(est, feature_names=feature_names)
        except Exception:
            tree_text = None
    elif a == 'random forest':
        tree_model_obj = est.estimators_[0]

    if hasattr(est, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': est.feature_importances_
        }).sort_values('Importance', ascending=False).reset_index(drop=True)

    # Calculate metrics
    classes_list = range(len(le.classes_))
    metrics = {
        'Accuracy': float(accuracy_score(yte, pred)),
        'Precision': float(precision_score(yte, pred, average='weighted', zero_division=0)),
        'Precision (Macro)': float(precision_score(yte, pred, average='macro', zero_division=0)),
        'Recall': float(recall_score(yte, pred, average='weighted', zero_division=0)),
        'Recall (Macro)': float(recall_score(yte, pred, average='macro', zero_division=0)),
        'F1': float(f1_score(yte, pred, average='weighted', zero_division=0)),
        'F1 (Macro)': float(f1_score(yte, pred, average='macro', zero_division=0)),
        'Kappa': float(cohen_kappa_score(yte, pred)),
        'Confusion Matrix': confusion_matrix(yte, pred, labels=classes_list),
        'Report': classification_report(
            yte, pred, labels=classes_list,
            target_names=[str(c) for c in le.classes_],
            output_dict=True, zero_division=0
        )
    }

    # ROC-AUC calculation
    if proba is not None:
        try:
            if len(le.classes_) == 2:
                metrics['AUC'] = float(roc_auc_score(yte, proba[:, 1]))
            else:
                metrics['AUC'] = float(roc_auc_score(yte, proba, multi_class='ovr', labels=classes_list))
        except Exception:
            metrics['AUC'] = None
    else:
        metrics['AUC'] = None

    return {
        'metrics': metrics,
        'classes': le.classes_,
        'tree': tree_text,
        'tree_model': tree_model_obj,
        'feature_names': feature_names,
        'importances': importances,
        'algorithm': algorithm,
        'target': target,
        'proba': proba,
        'pred': pred,
        'y_true': yte,
        'train_samples': len(Xtr),
        'test_samples': len(Xte),
        'model_pipeline': model
    }
