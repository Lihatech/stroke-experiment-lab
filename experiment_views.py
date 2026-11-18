"""Streamlit controls for hyperparameter search and saved experiment comparison."""

import json
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from experiment import DEFAULT_BASE_MODELS, META_MODELS, metric_row
from tuning import METRICS, candidate_count, parameter_choices, parse_values, role_label


def display_value(value):
    return json.dumps(value) if not isinstance(value, str) else value


def tuning_controls(roles, model_label):
    with st.expander("Hyperparameter tuning and model settings"):
        mode = st.radio("Run mode", ["Without tuning", "With tuning", "With and without tuning"],
                        horizontal=True,
                        help="The paired mode saves two runs using the same chosen models, predictors, and folds.")
        chosen = st.multiselect("Models to configure", list(roles), format_func=role_label,
                                placeholder="Choose base models or combiners by role")
        search = mode != "Without tuning"
        config = {"method": "Random search", "metric": "Average precision",
                  "budget": 5, "folds": 3, "seed": 42, "profiles": {}}
        if search:
            if not chosen:
                st.caption("Select a model to open its tuning settings.")
            a, b, c = st.columns(3)
            config["method"] = a.selectbox("Search method", ["Random search", "Grid search"],
                                           help="Grid tries every combination. Random samples distinct combinations.")
            config["metric"] = b.selectbox(
                "Optimize for", list(METRICS),
                help="This metric picks the winning settings. Average precision measures how well the model ranks rare stroke cases.",
            )
            config["folds"] = c.number_input("Search folds", min_value=2, value=3, step=1)
            a, b = st.columns(2)
            if config["method"] == "Random search":
                config["budget"] = a.number_input("Trials per model per fit", min_value=1, value=5, step=1)
            config["seed"] = b.number_input("Search seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
    invalid = set()
    for role in chosen:
        model = roles[role]
        label = role_label(role)
        profile = {"fixed": {}, "search": {}}
        with st.expander(f"{role[0]} · {model_label(model)} settings", expanded=len(chosen) == 1):
            advanced = st.checkbox("Show advanced estimator parameters", key=f"{label}:advanced",
                                   help="Show additional scalar settings supported by the installed estimator.")
            spec = parameter_choices(model, advanced)
            fixed = st.multiselect("Parameters to set", list(spec), key=f"{label}:fixed",
                                   help="These settings are also used in the untuned baseline.")
            searched = []
            if search:
                search_key = f"{label}:search"
                suggested = "learning_rate" if model in {"MLP", "Small MLP", "Tabular ResNet"} else next(iter(spec))
                if not st.session_state.get(search_key):
                    st.button(f"Use suggested parameter: {suggested}", key=f"{label}:suggest",
                              on_click=st.session_state.update, args=({search_key: [suggested]},))
                searched = st.multiselect("Parameters to tune", list(spec), key=search_key,
                                            help="A parameter can have a fixed baseline value and a separate list of trial values.")
            for parameter in dict.fromkeys([*fixed, *searched]):
                cols = st.columns(2) if parameter in fixed and parameter in searched else [st]
                if parameter in fixed:
                    field = cols[0]
                    raw = field.text_input(f"{parameter} · Fixed value", display_value(spec[parameter][0]),
                                           key=f"{label}:{parameter}:fixed-value")
                    try:
                        values = parse_values(model, parameter, raw)
                        if len(values) != 1:
                            raise ValueError("Enter one fixed value.")
                        profile["fixed"][parameter] = values[0]
                    except (ValueError, OverflowError) as exc:
                        field.error(str(exc))
                        invalid.add(label)
                if parameter in searched:
                    field = cols[-1]
                    raw = field.text_input(f"{parameter} · Trial values",
                                           ", ".join(map(display_value, spec[parameter])),
                                           help="Enter a comma-separated list. For a numeric range use start:stop:count, "
                                                "for example 0.01:0.1:5. Endpoints are included.",
                                           key=f"{label}:{parameter}:search-values")
                    try:
                        profile["search"][parameter] = parse_values(model, parameter, raw)
                    except (ValueError, OverflowError) as exc:
                        field.error(str(exc))
                        invalid.add(label)
            if profile["search"] and label not in invalid:
                n = candidate_count(profile["search"], config["method"], config["budget"])
                st.caption(f"{n:,} candidate settings per fit · {config['folds']} search folds")
        if profile["fixed"] or profile["search"]:
            config["profiles"][label] = profile
    return mode, config, invalid


def saved_metrics(folder, predictions=None):
    metrics = pd.read_csv(folder / "metrics.csv")
    if "Accuracy" in metrics:
        return metrics
    if predictions is None:
        predictions = pd.read_csv(folder / "predictions.csv")
    keys = ["Feature set", "Model", "Fold"]
    return pd.DataFrame([
        {**dict(zip(keys, group)), **metric_row(rows.stroke, rows.score)}
        for group, rows in predictions.groupby(keys, sort=False)
    ])


def render_trials(folder, settings, section):
    path = folder / "tuning_trials.csv"
    if not path.is_file():
        return
    config = settings["tuning"]
    with st.expander("Hyperparameter trial results"):
        st.caption(f"Mean inner score averages {config['metric']} over {config['folds']} search folds. "
                   "Selected marks the winning settings for that fit.")
        trials = pd.read_csv(path)
        view = st.radio("Show trials", ["Selected settings", "All candidates"], horizontal=True,
                        key=f"trials-view:{folder.name}:{section}")
        scope = st.selectbox("Fits to show", ["Outer evaluation fits", "All fits"],
                             help="Outer evaluation fits produce the headline results. All fits also includes the searches used to train the stacks.",
                             key=f"trials-scope:{folder.name}:{section}")
        if scope == "Outer evaluation fits":
            trials = trials[trials["Evaluation fit"]]
        st.dataframe(trials[trials.Selected] if view == "Selected settings" else trials, hide_index=True)
        st.download_button("Download tuning trials", path.read_bytes(), path.name,
                           key=f"trials-download:{folder.name}:{section}")


def compare_runs(folders, display_name):
    st.header("Compare experiments")
    st.caption("Select two or more runs to compare the same model and feature set.")
    if len(folders) < 2:
        st.info("Save at least two experiments to compare them here.")
        return
    folders = {folder.name: folder for folder in folders}
    settings = {key: json.loads((folder / "settings.json").read_text(encoding="utf-8"))
                for key, folder in folders.items()}
    names = {key: display_name(value) for key, value in settings.items()}
    counts_by_name = Counter(names.values())
    for key, name in names.copy().items():
        if counts_by_name[name] > 1:
            names[key] = f"{name} · {key.rsplit('-', 1)[-1]}"
    chosen = st.multiselect("Experiments", list(folders), format_func=names.get, key="compare-runs",
                            placeholder="Select saved experiments")
    if len(chosen) < 2:
        return

    a, b, c = st.columns(3)
    reference = a.selectbox("Reference experiment", chosen, format_func=names.get,
                            help="Differences subtract this run's score from each other run's score.")
    group = b.selectbox("Feature set", ["Selected predictors", "All predictors"])
    metrics = {key: saved_metrics(folders[key]) for key in chosen}
    available = sorted(set().union(*(set(table.Model) for table in metrics.values())))
    default = next((model for model in ("Final stack", "Stack", "MLP") if model in available), available[0])
    model = c.selectbox("Model", available, index=available.index(default),
                        help="Models missing from a run appear as blank cells.")

    fields = ["dataset_sha256", "outer_folds", "seed", "threshold"]
    comparable = all(all(settings[k].get(field) == settings[reference].get(field)
                         and settings[k].get(field) is not None for field in fields) for k in chosen)
    if comparable:
        splits = [pd.read_csv(folders[key] / "splits.csv").sort_values("id").reset_index(drop=True)
                  if (folders[key] / "splits.csv").is_file() else None for key in chosen]
        comparable = all(split is not None and split.equals(splits[0]) for split in splits)
    if not comparable:
        st.warning("These runs have different or unverified evaluation settings. Compare scores descriptively.")

    display = {}
    counts = ["TN", "FP", "FN", "TP"]
    for key, table in metrics.items():
        rows = table[(table["Feature set"] == group) & (table["Model"] == model)].drop(
            columns=["Feature set", "Model", "Fold"])
        summary = rows.drop(columns=counts, errors="ignore").mean(numeric_only=True)
        display[names[key]] = pd.concat([summary, rows[counts].sum(min_count=1)])
    result = pd.DataFrame(display)
    if comparable and st.checkbox("Show differences from reference", value=True):
        for key in chosen:
            if key != reference:
                result[f"Δ {names[key]}"] = result[names[key]] - result[names[reference]]
    result.index.name = "Metric"
    st.dataframe(result.round(4), width="stretch")
    st.download_button("Download comparison CSV", result.to_csv(), "experiment-comparison.csv")

    with st.expander("Compare experiment settings", expanded=True):
        settings_table = {}
        for key in chosen:
            run = settings[key]
            tuning = run.get("tuning") or {}
            settings_table[names[key]] = {
                "Tuning": tuning.get("method", "On") if tuning.get("enabled") else "Off",
                "Configured models": ", ".join(tuning.get("profiles", {})),
                "Parameter settings": json.dumps(tuning["profiles"], ensure_ascii=False) if tuning.get("profiles") else "",
                "Search folds": str(tuning["folds"]) if tuning.get("enabled") else "",
                "Search seed": str(tuning["seed"]) if tuning.get("enabled") else "",
                "Search metric": tuning.get("metric", "") if tuning.get("enabled") else "",
                "Search budget": (("All combinations" if tuning["method"] == "Grid search" else
                                   str(tuning.get("budget", ""))) if tuning.get("enabled") else ""),
                "Selector": run["method"], "Predictors kept": str(run.get("feature_count", len(run["selected"]))),
                "Candidate predictors": ", ".join(run["selected"]),
                "Stack 1": ", ".join(run.get("stack_base_models", DEFAULT_BASE_MODELS)),
                "Stack 1 combiner": run.get("stack_meta_model", META_MODELS[0]),
                "Stack 2": ", ".join(run.get("stack2_base_models", [])),
                "Stack 2 combiner": run.get("stack2_combiner", ""),
                "Final combiner": run.get("final_combiner", ""),
                "Cutoff": str(run.get("threshold", "")),
                "Runtime (s)": str(run.get("elapsed_seconds", "")),
            }
        config_table = pd.DataFrame(settings_table)
        st.dataframe(config_table, width="stretch")
        st.download_button("Download settings comparison CSV", config_table.to_csv(),
                           "experiment-settings-comparison.csv")

    view = st.selectbox("Comparison chart", ["Precision–recall curves", "ROC curves", "Confusion matrices"])
    if view == "Confusion matrices":
        for offset in range(0, len(chosen), 3):
            cols = st.columns(len(chosen[offset:offset + 3]))
            for col, key in zip(cols, chosen[offset:offset + 3]):
                col.write(names[key])
                row = result[names[key]]
                if row[counts].isna().any():
                    col.caption("This model has no results for the chosen feature set.")
                    continue
                axis_labels = ["No stroke", "Stroke"]
                plot = px.imshow([[row.TN, row.FP], [row.FN, row.TP]], x=axis_labels, y=axis_labels,
                                 text_auto=True, color_continuous_scale="Blues",
                                 labels={"x": "Predicted", "y": "Actual", "color": "Count"})
                col.plotly_chart(plot, key=f"compare-confusion:{key}")
        return

    figure = go.Figure()
    for key in chosen:
        rows = pd.read_csv(folders[key] / "predictions.csv", usecols=["Feature set", "Model", "stroke", "score"])
        rows = rows[(rows["Feature set"] == group) & (rows["Model"] == model)]
        if rows.empty:
            continue
        if view == "ROC curves":
            x, y, _ = roc_curve(rows.stroke, rows.score)
        else:
            y, x, _ = precision_recall_curve(rows.stroke, rows.score)
        figure.add_scatter(x=x, y=y, name=names[key], mode="lines")
    if view == "ROC curves":
        figure.update_layout(xaxis_title="False-positive rate", yaxis_title="True-positive rate")
    else:
        figure.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(figure, key="compare-curves")

