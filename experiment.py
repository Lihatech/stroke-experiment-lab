"""Train and evaluate experiments on the Mendeley NHANES stroke dataset."""

import hashlib
import io
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score, log_loss, cohen_kappa_score,
)
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from subset_search import HHO, ITERATIONS, POPULATION, PSO, search_subset
from storage import save_run_to_database

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "nhanes-stroke-v1.csv"
RUNS = ROOT / "runs" / "nhanes-v1"
SOURCE = "https://data.mendeley.com/datasets/xggs239bnw/1"
DATASET_SHA256 = "0b2ab231579aa7f2c93b5311b8acb3776335329995fd564880909b193ee6eddf"
FEATURES = {
    "gender": "Gender",
    "age": "Age group",
    "Race": "Race",
    "Marital status": "Marital status",
    "alcohol": "Alcohol",
    "smoke": "Smoking",
    "sleep disorder": "Sleep disorder",
    "Health Insurance": "Health insurance",
    "General health condition": "General health condition",
    "depression": "Depression",
    "sleep time": "Sleep time",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "high cholesterol": "High cholesterol",
    "Minutes sedentary activity": "Minutes of sedentary activity",
    "Coronary Heart Disease": "Coronary heart disease",
    "Body Mass Index": "Body mass index group",
    **{name: name for name in (
        "Waist Circumference", "Systolic blood pressure", "Diastolic blood pressure",
        "High-density lipoprotein", "Triglyceride", "Low-density lipoprotein",
        "Fasting Glucose", "Glycohemoglobin", "energy", "protein", "Carbohydrate",
        "Dietary fiber", "Total fat", "Total saturated fatty acids",
        "Total monounsaturated fatty acids", "Total polyunsaturated fatty acids",
        "Potassium", "Sodium",
    )},
}
CATEGORICAL = {
    "gender", "age", "Race", "Marital status", "alcohol", "smoke", "sleep disorder",
    "Health Insurance", "General health condition", "depression", "diabetes",
    "hypertension", "high cholesterol", "Coronary Heart Disease", "Body Mass Index",
}
NUMERIC = FEATURES.keys() - CATEGORICAL
DEMO_FEATURES = ["age", "Body Mass Index", "Fasting Glucose", "hypertension"]
INFO_GAIN = "Information Gain (binned)"
L1 = "L1 Logistic Regression"
PRESETS = ("As selected", "Mutual information", "ANOVA F-test", "Pearson correlation",
           INFO_GAIN, L1, PSO, HHO)
DEFAULT_BASE_MODELS = ("MLP", "Tabular ResNet")
BENCHMARK_MODELS = (*DEFAULT_BASE_MODELS, "Boosting", "Logistic regression")
PAPER_STACK2 = ("LightGBM", "AdaBoost", "XGBoost", "CatBoost")
BOOSTING_MODELS = (*PAPER_STACK2, "Boosting", "Gradient boosting")
ADDITIONAL_MODELS = ("Extra trees", "Gaussian Naive Bayes", "SVM (linear)", "SVM (RBF)", "SVM (sigmoid)")
STACK_BASE_OPTIONS = (*BENCHMARK_MODELS, "Random forest", "Decision tree", "KNN",
                      *PAPER_STACK2, "Gradient boosting", *ADDITIONAL_MODELS)
META_MODELS = ("Logistic regression", "Small MLP", "SVM (RBF)", "KNN",
               "Random forest", "Gradient boosting", "Decision tree", "Extra trees",
               "Gaussian Naive Bayes", "SVM (linear)", "SVM (sigmoid)",
               "Boosting", *PAPER_STACK2)
DEFAULT_STACK2 = PAPER_STACK2
EXPERIMENT_TZ = ZoneInfo("Asia/Kolkata")
SEED = 42
FOLDS = 3
EPOCHS = 25
THRESHOLD = 0.5


def external_classifier(name, labels, balance):
    try:
        if name == "LightGBM":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=60, num_leaves=15, n_jobs=1, verbosity=-1,
                                  class_weight="balanced" if balance else None, random_state=SEED)
        if name == "XGBoost":
            from xgboost import XGBClassifier
            positives = int(np.sum(labels))
            weight = (len(labels) - positives) / positives if balance else 1
            return XGBClassifier(n_estimators=60, max_depth=4, tree_method="hist", n_jobs=1,
                                 scale_pos_weight=weight, random_state=SEED)
        if name == "CatBoost":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=60, depth=4, thread_count=1, verbose=False,
                                      allow_writing_files=False, random_seed=SEED,
                                      auto_class_weights="Balanced" if balance else None)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"{name} could not load. Install the project requirements in its Python environment: {exc}") from exc
    raise ValueError(f"Unknown classifier: {name}")


def svm_classifier(name, balance=False, options=None):
    kernel = name.removeprefix("SVM (").removesuffix(")").lower()
    classifier = SVC(kernel=kernel, class_weight="balanced" if balance else None)
    classifier.set_params(**(options or {}))
    return CalibratedClassifierCV(make_pipeline(StandardScaler(), classifier), cv=3, ensemble=False)


def additional_classifier(name, balance=False):
    weight = "balanced" if balance else None
    if name.startswith("SVM ("):
        return svm_classifier(name, balance)
    if name == "Gaussian Naive Bayes":
        return GaussianNB()
    if name == "Extra trees":
        return ExtraTreesClassifier(n_estimators=60, min_samples_leaf=5, n_jobs=1,
                                    class_weight=weight, random_state=SEED)
    if name == "Gradient boosting":
        return GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=SEED)
    raise ValueError(f"Unknown classifier: {name}")


def read_dataset(contents):
    fingerprint = hashlib.sha256(contents).hexdigest()
    frame = pd.read_csv(io.BytesIO(contents))
    frame.columns = frame.columns.str.strip()
    if not frame.columns.is_unique or set(frame.columns) != {"stroke", *FEATURES}:
        raise ValueError("The columns do not match the Mendeley NHANES stroke dataset.")
    if len(frame) != 4603 or frame["stroke"].value_counts().to_dict() != {0: 4241, 1: 362}:
        raise ValueError("Expected 4,603 records with 362 stroke cases and 4,241 controls.")
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name]).any() or frame[name].isna().all():
            raise ValueError(f"Invalid values in {name}.")
    if fingerprint != DATASET_SHA256:
        raise ValueError("The file differs from the pinned Mendeley version 1 CSV. Upload the original file.")
    # The release has no participant ID. Use a 1-based source row number for audit joins.
    frame.insert(0, "id", np.arange(1, len(frame) + 1))
    return frame, fingerprint


def select_features(train, candidates, method, feature_count=5):
    top_k = min(feature_count, len(candidates))
    if method == "As selected" or top_k == len(candidates):
        return list(candidates)
    numeric = [name for name in candidates if name in NUMERIC]
    discrete = [name for name in candidates if name not in NUMERIC]
    if method in {PSO, HHO}:
        folds = []
        for fit_idx, val_idx in split_folds(train):
            fit, val = train.iloc[fit_idx], train.iloc[val_idx]
            encoder = make_preprocessor(candidates, scale=True).fit(fit)
            columns = feature_columns(encoder, numeric, discrete)
            folds.append((encoder.transform(fit), fit.stroke, encoder.transform(val), val.stroke, columns))
        return search_subset(method, folds, candidates, top_k, SEED)
    if method == L1:
        encoder = make_preprocessor(candidates, scale=True).fit(train)
        sklearn_version = tuple(map(int, version("scikit-learn").split(".")[:2]))
        regularization = {"l1_ratio": 1.0} if sklearn_version >= (1, 8) else {"penalty": "l1"}
        model = LogisticRegression(solver="liblinear", C=0.25, class_weight="balanced",
                                   random_state=SEED, **regularization)
        model.fit(encoder.transform(train), train.stroke)
        columns = feature_columns(encoder, numeric, discrete)
        scores = {name: np.abs(model.coef_[0, columns[name]]).max() for name in candidates}
    elif method == INFO_GAIN:
        scores = {}
        for name in candidates:
            values = train[name]
            if name in NUMERIC:
                values = values.fillna(values.median())
                edges = np.unique(values.quantile([0, .2, .4, .6, .8, 1]))
                values = np.searchsorted(edges[1:-1], values)
            else:
                values = values.fillna(values.mode().iloc[0])
            scores[name] = mutual_info_score(train.stroke, values)
    elif method in {"ANOVA F-test", "Pearson correlation"}:
        # Score indicators separately and take the largest score per original variable.
        scores = {}
        for name in candidates:
            values = train[[name]]
            if name in NUMERIC:
                encoded = SimpleImputer(strategy="median").fit_transform(values)
            else:
                values = values.astype("string").fillna("Missing").astype(str)
                encoded = OneHotEncoder(sparse_output=False).fit_transform(values)
            encoded = encoded[:, np.ptp(encoded, axis=0) > 0]
            if encoded.shape[1] == 0:
                scores[name] = 0.0
                continue
            if method == "ANOVA F-test":
                with np.errstate(divide="ignore", invalid="ignore"):
                    stats, _ = f_classif(encoded, train["stroke"])
            else:
                target = train["stroke"].to_numpy()
                centered = encoded - encoded.mean(axis=0)
                target = target - target.mean()
                stats = np.abs(centered.T @ target) / (np.linalg.norm(centered, axis=0) * np.linalg.norm(target))
            scores[name] = float(np.nan_to_num(stats, nan=0).max())
    else:
        encoder = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", make_pipeline(
                SimpleImputer(strategy="most_frequent"), OrdinalEncoder()
            ), discrete),
        ])
        values = encoder.fit_transform(train[candidates])
        stats = mutual_info_classif(
            values, train["stroke"], discrete_features=[False] * len(numeric) + [True] * len(discrete),
            random_state=SEED,
        )
        scores = dict(zip(numeric + discrete, stats))
    return sorted(candidates, key=lambda name: (-scores[name], name))[:top_k]


def make_preprocessor(features, scale):
    numeric = [name for name in features if name in NUMERIC]
    categories = [name for name in features if name not in NUMERIC]
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale:
        numeric_steps.append(StandardScaler())
    return ColumnTransformer([
        ("numeric", make_pipeline(*numeric_steps), numeric),
        ("categorical", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), categories),
    ])


def feature_columns(encoder, numeric, discrete):
    columns = {name: np.array([i]) for i, name in enumerate(numeric)}
    offset = len(numeric)
    if discrete:
        categories = encoder.named_transformers_["categorical"].named_steps["onehotencoder"].categories_
        for name, values in zip(discrete, categories):
            columns[name] = np.arange(offset, offset + len(values))
            offset += len(values)
    return columns


def preprocess(train, validation, features, scale):
    transformer = make_preprocessor(features, scale)
    return transformer.fit_transform(train), transformer.transform(validation)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, values):
        return values + self.layers(values)


def fit_neural(name, x_train, y_train, x_predict, seed, *, width=None, dropout=0.1,
               learning_rate=0.001, weight_decay=0.01, batch_size=128, epochs=EPOCHS, pos_weight=None):
    torch.manual_seed(seed)
    width = width or (8 if name == "Small MLP" else 32)
    if name == "MLP":
        model = nn.Sequential(nn.Linear(x_train.shape[1], width), nn.ReLU(), nn.Dropout(dropout),
                              nn.Linear(width, width // 2), nn.ReLU(), nn.Linear(width // 2, 1))
    elif name == "Tabular ResNet":
        model = nn.Sequential(nn.Linear(x_train.shape[1], width), ResidualBlock(width, dropout),
                              ResidualBlock(width, dropout), nn.LayerNorm(width), nn.ReLU(), nn.Linear(width, 1))
    elif name == "Small MLP":
        model = nn.Sequential(nn.Linear(x_train.shape[1], width), nn.ReLU(), nn.Linear(width, 1))
    else:
        raise ValueError(f"Unknown neural model: {name}")
    x = torch.as_tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weight = ((len(y) - y.sum()) / y.sum() if name != "Small MLP" else None)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)) if pos_weight is not None else weight)
    history = []
    for _ in range(epochs):
        total = 0.0
        for batch in torch.randperm(len(y)).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]).flatten(), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(y))
    model.eval()
    with torch.inference_mode():
        scores = torch.sigmoid(model(torch.as_tensor(x_predict, dtype=torch.float32))).flatten().numpy()
    return scores, history


def split_folds(frame):
    return StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(frame, frame["stroke"])


def fit_base(train, evaluation, features, seed, models, options=None):
    options = options or {}
    scores, history = {}, {}
    labels = train["stroke"]
    if any(name in models for name in (*DEFAULT_BASE_MODELS, "Logistic regression", "KNN")):
        x, x_eval = preprocess(train, evaluation, features, scale=True)
        for i, name in enumerate(DEFAULT_BASE_MODELS):
            if name in models:
                scores[name], history[name] = fit_neural(name, x, labels.to_numpy(), x_eval, seed + i,
                                                          **options.get(name, {}))
        if "Logistic regression" in models:
            linear = LogisticRegression(class_weight="balanced", max_iter=500, random_state=SEED)
            linear.set_params(**options.get("Logistic regression", {}))
            scores["Logistic regression"] = linear.fit(x, labels).predict_proba(x_eval)[:, 1]
        if "KNN" in models:
            knn = KNeighborsClassifier(n_neighbors=15, weights="distance")
            knn.set_params(**options.get("KNN", {}))
            scores["KNN"] = knn.fit(x, labels).predict_proba(x_eval)[:, 1]
    tree_models = {
        "Boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, class_weight="balanced", random_state=SEED),
        "Random forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
        "Decision tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=SEED),
    }
    extra = tuple(name for name in models if name not in tree_models
                  and name in (*PAPER_STACK2, "Gradient boosting", *ADDITIONAL_MODELS))
    if tree_models.keys() & set(models) or extra:
        x, x_eval = preprocess(train, evaluation, features, scale=False)
        with threadpool_limits(limits=1):
            for name, model in tree_models.items():
                if name in models:
                    model.set_params(**options.get(name, {}))
                    scores[name] = model.fit(x, labels).predict_proba(x_eval)[:, 1]
            for name in extra:
                if name in PAPER_STACK2 and name != "AdaBoost":
                    classifier = external_classifier(name, labels, balance=True)
                else:
                    classifier = (svm_classifier(name, True, options.get(name)) if name.startswith("SVM (")
                                  else additional_classifier(name, balance=True))
                if not name.startswith("SVM ("):
                    classifier.set_params(**options.get(name, {}))
                if name == "LightGBM":
                    scores[name] = classifier.fit(pd.DataFrame(x), labels).predict_proba(pd.DataFrame(x_eval))[:, 1]
                else:
                    scores[name] = classifier.fit(x, labels).predict_proba(x_eval)[:, 1]
    return scores, history


def base_oof(train, candidates, method, models, feature_count):
    oof = np.full((len(train), len(models)), np.nan)
    for fold, (train_idx, held_idx) in enumerate(split_folds(train), 1):
        inner = train.iloc[train_idx]
        features = select_features(inner, candidates, method, feature_count)
        inner_scores, _ = fit_base(inner, train.iloc[held_idx], features, SEED + fold, models)
        oof[held_idx] = np.column_stack([inner_scores[name] for name in models])
    return oof


def combine_scores(method, train_scores, labels, evaluation_scores, options=None):
    options = options or {}
    if method == "Small MLP":
        scaler = StandardScaler().fit(train_scores)
        scores, _ = fit_neural(method, scaler.transform(train_scores), labels.to_numpy(),
                               scaler.transform(evaluation_scores), SEED, **options)
        return scores
    estimators = {
        "Logistic regression": LogisticRegression(random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Random forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=10, random_state=SEED),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=SEED),
        "Boosting": HistGradientBoostingClassifier(max_leaf_nodes=15, random_state=SEED),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=SEED),
    }
    estimator = estimators.get(method)
    if estimator is None:
        estimator = (external_classifier(method, labels, balance=False) if method in PAPER_STACK2
                     else additional_classifier(method))
    if method.startswith("SVM ("):
        estimator = svm_classifier(method, options=options)
    else:
        estimator.set_params(**options)
    if method == "KNN":
        estimator = make_pipeline(StandardScaler(), estimator)
    if method == "LightGBM":
        train_scores, evaluation_scores = pd.DataFrame(train_scores), pd.DataFrame(evaluation_scores)
    with threadpool_limits(limits=1):
        return estimator.fit(train_scores, labels).predict_proba(evaluation_scores)[:, 1]


def predict_stacks(train, oof, base_scores, models, groups, combiners):
    results = []
    for group, combiner in zip(groups, combiners):
        columns = [models.index(name) for name in group]
        results.append(combine_scores(combiner, oof[:, columns], train.stroke, base_scores[:, columns]))
    return np.column_stack(results)


def fit_comparison(train, evaluation, candidates, method, base_models, meta_model, feature_count=5,
                   *, stack2_models=None, stack2_combiner=META_MODELS[0], final_combiner=META_MODELS[0]):
    groups = [base_models]
    combiners = [meta_model]
    if stack2_models is not None:
        groups.append(stack2_models)
        combiners.append(stack2_combiner)
    members = tuple(dict.fromkeys((*base_models, *(stack2_models or ()))))
    oof = base_oof(train, candidates, method, members, feature_count)
    features = select_features(train, candidates, method, feature_count)
    models = tuple(dict.fromkeys((*BENCHMARK_MODELS, *members)))
    scores, history = fit_base(train, evaluation, features, SEED, models)
    base_scores = np.column_stack([scores[name] for name in members])
    scores["Base-model average"] = base_scores.mean(axis=1)
    stacks = predict_stacks(train, oof, base_scores, members, groups, combiners)
    if stack2_models is None:
        scores["Stack"] = stacks[:, 0]
        return scores, features, history

    # Refit both full stacks inside every final-combiner training fold.
    # A held-out row is excluded from both the base and intermediate fits.
    stack_oof = np.full((len(train), 2), np.nan)
    for fit_idx, held_idx in split_folds(train):
        inner, held = train.iloc[fit_idx], train.iloc[held_idx]
        inner_oof = base_oof(inner, candidates, method, members, feature_count)
        inner_features = select_features(inner, candidates, method, feature_count)
        inner_scores, _ = fit_base(inner, held, inner_features, SEED, members)
        held_scores = np.column_stack([inner_scores[name] for name in members])
        stack_oof[held_idx] = predict_stacks(inner, inner_oof, held_scores, members, groups, combiners)
    scores["Stack 1"], scores["Stack 2"] = stacks.T
    scores["Final stack"] = combine_scores(final_combiner, stack_oof, train.stroke, stacks)
    return scores, features, history


def metric_row(labels, scores):
    predicted = scores >= THRESHOLD
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "Average precision": average_precision_score(labels, scores),
        "ROC AUC": roc_auc_score(labels, scores),
        "Accuracy": accuracy_score(labels, predicted),
        "Stroke recall": recall_score(labels, predicted),
        "Stroke precision": precision_score(labels, predicted, zero_division=0),
        "Specificity": tn / (tn + fp),
        "Negative predictive value": tn / (tn + fn) if tn + fn else np.nan,
        "F1": f1_score(labels, predicted, zero_division=0),
        "Balanced accuracy": balanced_accuracy_score(labels, predicted),
        "MCC": matthews_corrcoef(labels, predicted),
        "Cohen's kappa": cohen_kappa_score(labels, predicted),
        "Jaccard (stroke)": tp / (tp + fp + fn),
        "Hamming loss": (fp + fn) / len(labels),
        "Log loss": log_loss(labels, scores, labels=[0, 1]),
        "Brier score": brier_score_loss(labels, scores),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def development_split(frame):
    return train_test_split(frame, test_size=0.20, stratify=frame.stroke, random_state=SEED)


def run_experiment(frame, fingerprint, selected, method="As selected", progress=None, *,
                   base_models=DEFAULT_BASE_MODELS, meta_model=META_MODELS[0], experiment_name="",
                   feature_count=5, stack2_models=None, stack2_combiner=META_MODELS[0],
                   final_combiner=META_MODELS[0], tuning=None, paired_id=None):
    if not selected or len(set(selected)) != len(selected) or set(selected) - FEATURES.keys():
        raise ValueError("Choose one or more valid predictors.")
    if method not in PRESETS:
        raise ValueError("Unknown feature-selection method.")
    if not isinstance(feature_count, int) or not 1 <= feature_count <= len(FEATURES):
        raise ValueError(f"Features to keep must be between 1 and {len(FEATURES)}.")
    if (len(base_models) < 2 or len(base_models) != len(set(base_models))
            or set(base_models) - set(STACK_BASE_OPTIONS)):
        raise ValueError("Choose at least two distinct stack base models.")
    if meta_model not in META_MODELS:
        raise ValueError("Choose a supported stack combiner.")
    if stack2_models is not None:
        if (len(set(stack2_models)) < 2 or len(stack2_models) != len(set(stack2_models))
                or set(stack2_models) - set(STACK_BASE_OPTIONS)):
            raise ValueError("Choose at least two distinct Stack 2 base models.")
        if stack2_combiner not in META_MODELS or final_combiner not in META_MODELS:
            raise ValueError("Choose supported Stack 2 and final combiners.")
    now = datetime.now(timezone.utc)
    experiment_name = experiment_name.strip()
    if len(experiment_name) > 100:
        raise ValueError("Use at most 100 characters for the experiment name.")
    if not experiment_name:
        experiment_name = f"{method} · {now.astimezone(EXPERIMENT_TZ):%Y-%m-%d %H:%M:%S}"
    from tuning import evaluate_configured, model_roles, validate_config

    roles, upstream = model_roles(base_models, meta_model, stack2_models, stack2_combiner, final_combiner)
    validate_config(tuning, roles)
    trials = []
    started = perf_counter()
    torch.set_num_threads(1)
    development, holdout = development_split(frame)
    groups = {"All predictors": (list(FEATURES), "As selected"),
              "Selected predictors": (selected, method)}
    predictions, metrics, choices, losses, summaries, assignments = [], [], [], [], [], []
    total_steps = FOLDS * len(groups)
    for fold, (train_idx, val_idx) in enumerate(split_folds(development), 1):
        train, val = development.iloc[train_idx], development.iloc[val_idx]
        assignments.extend({"id": row_id, "partition": "development", "validation_fold": fold}
                           for row_id in val["id"])
        for offset, (group, (candidates, selector)) in enumerate(groups.items()):
            if progress:
                progress(((fold - 1) * len(groups) + offset) / total_steps, f"Fold {fold} of {FOLDS} · {group}")
            if tuning and tuning["profiles"]:
                def report_search(role, model, attempt, total):
                    if progress:
                        progress(((fold - 1) * len(groups) + offset) / total_steps,
                                 f"Fold {fold} of {FOLDS} · {group} · {role}: {model} · Trial {attempt}/{total}")

                scores, features, history = evaluate_configured(
                    train, val, candidates, selector, feature_count, roles, upstream, tuning,
                    trials, fold, group, report_search,
                )
            else:
                scores, features, history = fit_comparison(
                    train, val, candidates, selector, base_models, meta_model, feature_count,
                    stack2_models=stack2_models, stack2_combiner=stack2_combiner,
                    final_combiner=final_combiner,
                )
            details = {"Feature set": group, "Fold": fold}
            choices.extend({**details, "Feature": name} for name in features)
            summaries.append({
                **details, "Training rows": len(train), "Training strokes": int(train["stroke"].sum()),
                "Validation rows": len(val), "Validation strokes": int(val["stroke"].sum()),
                "Training missing inputs": int(train[features].isna().sum().sum()),
                "Validation inputs imputed": int(val[features].isna().sum().sum()),
            })
            for model, values in scores.items():
                metrics.append({**details, "Model": model, **metric_row(val["stroke"], values)})
                predictions.extend({
                    **details, "Model": model, "id": row_id, "stroke": label, "score": score,
                } for row_id, label, score in zip(val["id"], val["stroke"], values))
            for model, values in history.items():
                losses.extend({**details, "Model": model, "Epoch": epoch, "Loss": loss}
                              for epoch, loss in enumerate(values, 1))
    assignments.extend({"id": row_id, "partition": "holdout", "validation_fold": ""}
                       for row_id in holdout["id"])
    run_id = f"{now:%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    output = RUNS / run_id
    output.mkdir(parents=True)
    tables = {"predictions": predictions, "metrics": metrics, "features": choices,
              "learning": losses, "preprocessing": summaries, "splits": assignments}
    for filename, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"{filename}.csv", index=False)
    settings = {
        "run_id": run_id, "created_utc": now.isoformat(), "source": SOURCE,
        "experiment_name": experiment_name,
        "dataset_sha256": fingerprint, "selected": selected, "method": method,
        "feature_count": len(selected) if method == "As selected" else min(feature_count, len(selected)),
        "stack_base_models": list(base_models), "stack_meta_model": meta_model,
        "row_identifier": "1-based source data row number, excluding the header",
        "seed": SEED, "outer_folds": FOLDS, "stacking_folds": FOLDS,
        "epochs": EPOCHS, "threshold": THRESHOLD, "holdout_rows": len(holdout),
        "elapsed_seconds": round(perf_counter() - started, 2),
        "holdout_strokes": int(holdout["stroke"].sum()), "python": platform.python_version(),
        "packages": {name: version(name) for name in ("torch", "scikit-learn", "pandas", "numpy")},
    }
    if tuning:
        settings["tuning"] = tuning
    if paired_id:
        settings["paired_id"] = paired_id
    if trials:
        pd.DataFrame(trials).to_csv(output / "tuning_trials.csv", index=False)
    if method in {PSO, HHO}:
        settings["selector_search"] = {"population": POPULATION, "iterations": ITERATIONS,
                                       "objective": "Mean training-only CV average precision, logistic regression"}
    if stack2_models is not None:
        settings.update(stack2_base_models=list(stack2_models), stack2_combiner=stack2_combiner,
                        final_combiner=final_combiner)
    chosen_models = {*base_models, *(stack2_models or ()), meta_model}
    if stack2_models is not None:
        chosen_models.update((stack2_combiner, final_combiner))
    for package in ("LightGBM", "XGBoost", "CatBoost"):
        if package in chosen_models:
            settings["packages"][package.lower()] = version(package.lower())
    pending = output / "settings.tmp"
    pending.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    pending.replace(output / "settings.json")
    try:
        save_run_to_database(output)
    except Exception:
        settings["storage_warning"] = (
            "Database upload failed. This run is available only in the current app instance."
        )
        (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return run_id
