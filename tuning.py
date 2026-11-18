"""Model-level search using the same fit functions as the standard benchmark."""

import json
import math
from functools import lru_cache
from inspect import Parameter, signature
from itertools import product
from random import Random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import experiment as lab

METRICS = {
    "Average precision": average_precision_score,
    "ROC AUC": roc_auc_score,
    "F1": lambda y, score: f1_score(y, score >= lab.THRESHOLD, zero_division=0),
    "Balanced accuracy": lambda y, score: balanced_accuracy_score(y, score >= lab.THRESHOLD),
    "Accuracy": lambda y, score: accuracy_score(y, score >= lab.THRESHOLD),
}
NEURAL = dict(width=(16, 32, 64), learning_rate=(0.0003, 0.001, 0.003),
              weight_decay=(0.0, 0.01, 0.05), batch_size=(32, 64, 128), epochs=(15, 25, 40),
              pos_weight=(None, 1.0, 5.0, 10.0))
DROPOUT = dict(dropout=(0.0, 0.1, 0.25))
CLASS_WEIGHT = dict(class_weight=(None, "balanced"))
TREE = dict(max_depth=(3, 6, 12, None), min_samples_leaf=(2, 5, 10),
            min_samples_split=(2, 5, 10), max_features=(None, "sqrt", "log2", 0.5),
            criterion=("gini", "entropy", "log_loss"), min_impurity_decrease=(0.0, 0.001, 0.01),
            ccp_alpha=(0.0, 0.001, 0.01))
FOREST = dict(**TREE, **CLASS_WEIGHT, n_estimators=(50, 100, 200), bootstrap=(True, False))
BOOST = dict(n_estimators=(30, 60, 120), learning_rate=(0.03, 0.1, 0.3))
GB_TREE = dict(max_depth=(2, 4, 6), min_samples_leaf=(2, 5, 10), subsample=(0.7, 0.85, 1.0))
SVM = dict(C=(0.1, 1.0, 10.0), tol=(0.0001, 0.001, 0.01), **CLASS_WEIGHT)
SPECS = {
    "MLP": dict(**NEURAL, **DROPOUT),
    "Tabular ResNet": dict(**NEURAL, **DROPOUT),
    "Small MLP": dict(NEURAL, width=(4, 8, 16)),
    "Logistic regression": dict(C=(0.1, 1.0, 10.0), l1_ratio=(0.0, 0.5, 1.0),
        solver=("lbfgs", "liblinear", "saga", "newton-cg", "newton-cholesky", "sag"),
        max_iter=(100, 500, 1000), tol=(0.0001, 0.001, 0.01), fit_intercept=(True, False), **CLASS_WEIGHT),
    "KNN": dict(n_neighbors=(5, 15, 30), weights=("uniform", "distance"), p=(1, 2),
                algorithm=("auto", "brute", "ball_tree", "kd_tree"), leaf_size=(15, 30, 60)),
    "Gaussian Naive Bayes": dict(var_smoothing=(1e-11, 1e-9, 1e-7)),
    "SVM (linear)": SVM,
    "SVM (RBF)": dict(**SVM, gamma=("scale", "auto", 0.01, 0.1)),
    "SVM (sigmoid)": dict(**SVM, gamma=("scale", "auto", 0.01, 0.1), coef0=(-1.0, 0.0, 1.0)),
    "Decision tree": dict(**TREE, **CLASS_WEIGHT, splitter=("best", "random")),
    "Random forest": FOREST,
    "Extra trees": FOREST,
    "AdaBoost": BOOST,
    "Gradient boosting": dict(**BOOST, **GB_TREE, min_samples_split=(2, 5, 10),
                              max_features=(None, "sqrt", "log2", 0.5)),
    "Boosting": dict(max_iter=(50, 100, 200), learning_rate=BOOST["learning_rate"],
        max_leaf_nodes=(7, 15, 31), max_depth=TREE["max_depth"], min_samples_leaf=(5, 10, 20),
        l2_regularization=(0.0, 0.1, 1.0), max_bins=(63, 127, 255), **CLASS_WEIGHT),
    "LightGBM": dict(**BOOST, **CLASS_WEIGHT, num_leaves=(7, 15, 31), max_depth=(-1, 3, 6),
        min_child_samples=(5, 10, 20), colsample_bytree=(0.7, 0.85, 1.0),
        subsample=(0.7, 0.85, 1.0), subsample_freq=(0, 1, 3), reg_alpha=(0.0, 0.1, 1.0), reg_lambda=(0.0, 0.1, 1.0)),
    "XGBoost": dict(**BOOST, max_depth=(2, 4, 6), min_child_weight=(1.0, 3.0, 5.0),
        subsample=(0.7, 0.85, 1.0), colsample_bytree=(0.7, 0.85, 1.0),
        gamma=(0.0, 0.1, 1.0), reg_alpha=(0.0, 0.1, 1.0), reg_lambda=(0.0, 0.1, 1.0),
        scale_pos_weight=(1.0, 5.0, 10.0)),
    "CatBoost": dict(iterations=(30, 60, 120), learning_rate=BOOST["learning_rate"], depth=(3, 4, 6),
        l2_leaf_reg=(1.0, 3.0, 10.0), random_strength=(0.0, 1.0, 2.0),
        border_count=(32, 64, 128), auto_class_weights=(None, "Balanced", "SqrtBalanced")),
}


ESTIMATORS = {
    "Logistic regression": lab.LogisticRegression,
    "KNN": lab.KNeighborsClassifier,
    "Gaussian Naive Bayes": lab.GaussianNB,
    "SVM (linear)": lab.SVC, "SVM (RBF)": lab.SVC, "SVM (sigmoid)": lab.SVC,
    "Decision tree": lab.DecisionTreeClassifier, "Random forest": lab.RandomForestClassifier,
    "Extra trees": lab.ExtraTreesClassifier, "AdaBoost": lab.AdaBoostClassifier,
    "Gradient boosting": lab.GradientBoostingClassifier, "Boosting": lab.HistGradientBoostingClassifier,
}
# These settings are managed by the experiment, require structured input, or set up external I/O.
MANAGED = {
    "objective", "loss_function", "eval_metric", "metrics", "custom_loss", "custom_metric",
    "callbacks", "n_jobs", "thread_count", "device", "device_config", "task_type", "devices",
    "gpu_cat_features_storage", "gpu_ram_part", "used_ram_limit", "pinned_memory_size", "allow_writing_files",
    "model_file", "input_borders", "output_borders", "snapshot_file", "snapshot_interval", "train_dir",
    "save_snapshot", "logging_level", "verbose", "verbose_eval", "silent", "verbosity", "kernel",
    "probability", "categorical_features", "cat_features", "class_names", "classes_count",
    "interaction_constraints", "monotone_constraints", "monotonic_cst", "feature_weights",
    "text_features", "embedding_features", "ignored_features", "feature_names", "class_weights",
    "model_shrink_mode", "per_float_feature_quantization", "fixed_binary_splits",
    "feature_border_type", "boost_from_average", "early_stopping_rounds", "use_best_model", "eval_fraction",
    "random_state", "random_seed", "random_state_seed", "seed", "sample_weight", "estimator", "base_estimator",
    "importance_type", "predictor", "warm_start", "eval_set", "evals_result", "set_barrier",
}
# CatBoost exposes aliases that conflict with the app's canonical names.
CATBOOST_ALIASES = {"n_estimators", "num_boost_round", "num_trees", "max_depth",
                    "eta", "reg_lambda", "min_data_in_leaf", "random_state", "max_bin"}
ESTIMATOR_ERRORS = {"CatBoostError", "XGBoostError", "LightGBMError"}


@lru_cache(maxsize=None)
def advanced_parameters(model):
    estimator = ESTIMATORS.get(model)
    if model == "CatBoost":
        from catboost import CatBoostClassifier
        estimator = CatBoostClassifier
    elif model == "XGBoost":
        # The wrapper exposes the classifier-specific options and the inherited tree parameters.
        from xgboost import XGBClassifier
        defaults = XGBClassifier().get_params()
    elif model == "LightGBM":
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier
    if estimator is not None:
        defaults = {name: prop.default for name, prop in signature(estimator).parameters.items()
                    if prop.default is not Parameter.empty}
    elif model != "XGBoost":
        return {}
    exclude = MANAGED | (CATBOOST_ALIASES if model == "CatBoost" else set())
    return {
        name: value for name, value in defaults.items()
        if name not in exclude and name not in SPECS[model]
        and (value is None or type(value) in {bool, str, int, float})
        and (not isinstance(value, float) or math.isfinite(value))
    }


def parameter_choices(model, advanced):
    if not advanced:
        return SPECS[model]
    return {**SPECS[model], **{name: (value,) for name, value in advanced_parameters(model).items()}}


def model_roles(base_models, meta_model, stack2_models, stack2_combiner, final_combiner):
    roles = {("Stack 1 base", name): name for name in base_models}
    first = tuple(roles)
    stack1 = ("Stack 1 combiner", meta_model)
    roles[stack1] = meta_model
    upstream = {stack1: first}
    if stack2_models is not None:
        second = tuple(("Stack 2 base", name) for name in stack2_models)
        roles.update((key, key[1]) for key in second)
        stack2 = ("Stack 2 combiner", stack2_combiner)
        final = ("Final combiner", final_combiner)
        roles.update((key, key[1]) for key in (stack2, final))
        upstream.update({stack2: second, final: (stack1, stack2)})
    for name in lab.BENCHMARK_MODELS:
        roles["Reference model", name] = name
    return roles, upstream


def role_label(role):
    return " · ".join(role)


def valid_value(model, name, value):
    if name not in SPECS[model]:
        advanced = advanced_parameters(model)
        if name not in advanced:
            return False
        default = advanced[name]
        if default is None:
            return value is None or (
                type(value) in {bool, str, int, float} and (type(value) is not float or math.isfinite(value))
            )
        if type(default) is float:
            return type(value) in {int, float} and math.isfinite(value)
        return type(value) is type(default)
    template = SPECS[model][name]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return any(type(value) is type(option) and value == option for option in template)
    numbers = [option for option in template if type(option) in (int, float)]
    if not numbers or type(value) not in (float, int):
        return False
    if all(type(option) is int for option in numbers) and type(value) is not int:
        if name != "max_features":
            return False
    if not math.isfinite(value):
        return False
    if name in {"dropout", "l1_ratio"}:
        return 0 <= value <= 1
    if name in {"subsample", "colsample_bytree"} or (name == "max_features" and type(value) is float):
        return 0 < value <= 1
    if name == "max_bins":
        return 2 <= value <= 255
    if name == "depth":
        return 1 <= value <= 16
    if name == "coef0":
        return True
    if name == "max_depth" and model == "LightGBM":
        return value == -1 or value >= 1
    if name == "max_depth" and model == "XGBoost":
        return value >= 0
    if name in {"width", "min_samples_split", "max_leaf_nodes", "num_leaves"}:
        return value >= 2
    if name in {"C", "learning_rate", "tol", "var_smoothing", "pos_weight", "gamma"}:
        return value >= 0 if model == "XGBoost" and name == "gamma" else value > 0
    return value >= (0 if 0 in numbers else 1) or (value > 0 and any(type(n) is float for n in numbers))


def parse_values(model, name, raw):
    values = []
    for token in raw.split(","):
        token = token.strip()
        if ":" in token:
            start, stop, count = token.split(":")
            count = int(count)
            if count < 2:
                raise ValueError("A range needs at least two values.")
            if name in SPECS[model]:
                numbers = [option for option in SPECS[model][name] if type(option) in (int, float)]
                integer = bool(numbers) and all(type(option) is int for option in numbers)
            else:
                integer = type(advanced_parameters(model).get(name)) is int
            values.extend((int(round(v)) if integer else float(v)) for v in np.linspace(float(start), float(stop), count))
            continue
        try:
            value = json.loads(token.lower()) if token.lower() in {"true", "false", "null"} else json.loads(token)
        except ValueError:
            value = None if token.lower() == "none" else token
        values.append(value)
    if not all(valid_value(model, name, value) for value in values):
        raise ValueError("Enter valid values for this parameter.")
    return list(dict.fromkeys(values))


def validate_config(config, roles):
    if not config:
        return
    if (config["method"] not in {"Grid search", "Random search"} or config["metric"] not in METRICS
            or config["folds"] < 2 or config["budget"] < 1):
        raise ValueError("Choose a supported search method, metric, and positive search budget.")
    known = {role_label(role): model for role, model in roles.items()}
    if set(config["profiles"]) - known.keys():
        raise ValueError("A model in the tuning setup is absent from this run.")
    for label, profile in config["profiles"].items():
        model = known[label]
        for name, value in profile["fixed"].items():
            if not valid_value(model, name, value):
                raise ValueError(f"{label}: invalid fixed value for {name}.")
        for name, values in profile["search"].items():
            if not values or not all(valid_value(model, name, v) for v in values):
                raise ValueError(f"{label}: invalid search values for {name}.")
    if config["enabled"] and not any(p["search"] for p in config["profiles"].values()):
        raise ValueError("Choose at least one hyperparameter to search.")


def candidate_count(search, method, budget):
    count = math.prod(map(len, search.values()))
    return min(count, budget) if method == "Random search" else count


def candidates(search, config):
    values = tuple(search.values())
    if config["method"] == "Grid search":
        combinations = product(*values)
    else:
        total = math.prod(map(len, values))

        def decode(index):
            result = []
            for choices in reversed(values):
                index, remainder = divmod(index, len(choices))
                result.append(choices[remainder])
            return reversed(result)

        rng = Random(config["seed"])
        # Floyd sampling avoids materializing the grid, even for very large search spaces.
        sampled = set()
        for end in range(total - min(total, config["budget"]), total):
            index = rng.randrange(end + 1)
            sampled.add(end if index in sampled else index)
        combinations = (decode(index) for index in sorted(sampled))
    return (dict(zip(search, combination)) for combination in combinations)


def evaluate_configured(train, evaluation, features, selector, feature_count, roles, upstream, config,
                        trials, outer_fold, feature_group, report):
    """Fit a role using only its own training rows, including all upstream fits."""
    profiles = config["profiles"]
    frame = pd.concat([train, evaluation])
    train_rows, evaluation_rows = tuple(train.index), tuple(evaluation.index)
    source = {"fixed": {}, "search": {}}
    canonical = {}
    signatures = {}
    for role, model in roles.items():
        profile = profiles.get(role_label(role), source)
        if role not in upstream:
            signature = (model, json.dumps(profile, sort_keys=True))
            canonical[role] = signatures.setdefault(signature, role)

    @lru_cache(maxsize=None)
    def selected(rows):
        return lab.select_features(frame.loc[list(rows)], features, selector, feature_count)

    @lru_cache(maxsize=None)
    def partitions(rows, folds, seed):
        labels = frame.loc[list(rows), "stroke"]
        if labels.value_counts().min() < folds or labels.nunique() < 2:
            raise ValueError(f"{folds} folds require at least {folds} records of each class in the training partition.")
        ids = np.array(rows)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return tuple((tuple(ids[fit]), tuple(ids[held])) for fit, held in splitter.split(ids, labels))

    def predict(role, rows, held, seed=lab.SEED):
        key = canonical.get(role, role)
        return fit(key, rows, held, lab.SEED if key in upstream else seed)

    def inputs(role, rows, held):
        members = upstream[role]
        stacked = pd.DataFrame(index=list(rows), columns=range(len(members)), dtype=float)
        for fold, (inner, validation) in enumerate(partitions(rows, lab.FOLDS, lab.SEED), 1):
            stacked.loc[list(validation)] = np.column_stack(
                [predict(member, inner, validation, lab.SEED + fold)[0] for member in members]
            )
        held_scores = np.column_stack([predict(member, rows, held)[0] for member in members])
        return stacked.to_numpy(), held_scores

    def fit_once(role, rows, held, options, matrices=None, seed=lab.SEED):
        model = roles[role]
        training = frame.loc[list(rows)]
        if role in upstream:
            x, y = matrices if matrices is not None else inputs(role, rows, held)
            return lab.combine_scores(model, x, training.stroke, y, options), []
        scores, history = lab.fit_base(
            training, frame.loc[list(held)], selected(rows), seed, [model], {model: options},
        )
        return scores[model], history.get(model, [])

    @lru_cache(maxsize=None)
    def best_options(role, rows):
        profile = profiles.get(role_label(role), source)
        fixed, search = profile["fixed"], profile["search"]
        if not config["enabled"] or not search:
            return fixed
        folds = partitions(rows, config["folds"], config["seed"])
        matrices = [inputs(role, inner, held) for inner, held in folds] if role in upstream else [None] * len(folds)
        score_metric = METRICS[config["metric"]]
        total = candidate_count(search, config["method"], config["budget"])
        best_score, winner, last_error = -math.inf, None, ""
        search_id = len(trials) + 1
        for attempt, params in enumerate(candidates(search, config), 1):
            options = {**fixed, **params}
            report(*role, attempt, total)
            status = ""
            try:
                scores = [score_metric(frame.loc[list(held), "stroke"],
                                       fit_once(role, inner, held, options, matrix)[0])
                          for (inner, held), matrix in zip(folds, matrices)]
                mean_score = float(np.mean(scores))
                if not math.isfinite(mean_score):
                    raise ValueError("The search metric returned a non-finite score.")
            except Exception as exc:
                if not isinstance(exc, (ValueError, RuntimeError, TypeError)) and type(exc).__name__ not in ESTIMATOR_ERRORS:
                    raise
                last_error = status = str(exc)
                mean_score = None
            row = {"Outer fold": outer_fold, "Feature set": feature_group, "Role": role[0],
                   "Model": role[1], "Training rows": len(rows),
                   "Evaluation fit": rows == train_rows,
                   "Search fit": search_id,
                   "Trial": attempt, "Parameters": json.dumps(options, sort_keys=True),
                   "Mean inner score": mean_score, "Error": status, "Selected": False}
            trials.append(row)
            if mean_score is not None and mean_score > best_score:
                best_score, winner = mean_score, row
        if winner is None:
            raise ValueError(f"{role_label(role)} failed for every candidate. {last_error}")
        winner["Selected"] = True
        return json.loads(winner["Parameters"])

    @lru_cache(maxsize=None)
    def fit(role, rows, held, seed):
        options = best_options(role, rows)
        try:
            return fit_once(role, rows, held, options, seed=seed)
        except Exception as exc:
            if type(exc).__name__ not in ESTIMATOR_ERRORS:
                raise
            raise ValueError(f"{role_label(role)}: {exc}") from exc

    scores, history = {}, {}
    base_roles = tuple(dict.fromkeys(canonical[role] for role in canonical if role[0] != "Reference model"))
    # Preserve simple model names where the same algorithm is configured identically across roles.
    ordered = tuple(dict.fromkeys([*base_roles, *canonical.values()]))
    for role in ordered:
        label = role[1] if sum(item[1] == role[1] for item in ordered) == 1 else role_label(role)
        values, loss = predict(role, train_rows, evaluation_rows)
        scores[label] = values
        if loss:
            history[label] = loss
    scores["Base-model average"] = np.column_stack(
        [predict(role, train_rows, evaluation_rows)[0] for role in base_roles]
    ).mean(axis=1)
    labels = ["Stack 1", "Stack 2", "Final stack"] if len(upstream) == 3 else ["Stack"]
    for role, label in zip(upstream, labels):
        scores[label] = predict(role, train_rows, evaluation_rows)[0]
    return scores, selected(train_rows), history

