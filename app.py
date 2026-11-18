import hmac
import json
import os
from datetime import datetime
from uuid import uuid4

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from dataset_plots import render_dataset_plots
from experiment_views import compare_runs, render_trials, saved_metrics, tuning_controls
from tuning import model_roles, role_label, validate_config
from storage import database_configured, sync_runs_from_database

from experiment import (
    BOOSTING_MODELS, DATASET, DEFAULT_BASE_MODELS, DEFAULT_STACK2, DEMO_FEATURES, EXPERIMENT_TZ, FEATURES, META_MODELS,
    PRESETS, RUNS, SOURCE, STACK_BASE_OPTIONS, read_dataset, run_experiment,
)

STACK_PRESETS = {
    "Deep learning": DEFAULT_BASE_MODELS,
    "Hybrid": (*DEFAULT_BASE_MODELS, "Boosting"),
    "MDPI Stack 1 base models": ("Logistic regression", "Decision tree", "Random forest", "KNN"),
    "MDPI Stack 2 base models": DEFAULT_STACK2,
    "Boosting": BOOSTING_MODELS,
    "Tree ensembles": ("Random forest", "Extra trees", "Boosting", "Gradient boosting"),
    "Classical ML": ("Logistic regression", "Gaussian Naive Bayes", "KNN", "SVM (RBF)", "Random forest"),
    "All supported base models": STACK_BASE_OPTIONS,
}


def model_label(name):
    return "Histogram gradient boosting" if name == "Boosting" else name
SCORE_METRICS = {"Average precision", "ROC AUC", "Log loss", "Brier score"}
DEFAULT_METRICS = ["Average precision", "ROC AUC", "Accuracy", "Stroke recall",
                   "Stroke precision", "Specificity", "F1", "Balanced accuracy", "MCC"]


def stack_controls(label, default):
    preset_key = f"{label}-preset"
    models_key = f"{label}-models"
    if models_key not in st.session_state:
        st.session_state[models_key] = list(default)
    if preset_key not in st.session_state:
        st.session_state[preset_key] = next(name for name, models in STACK_PRESETS.items() if models == default)

    def apply_preset():
        preset = st.session_state[preset_key]
        if preset in STACK_PRESETS:
            st.session_state[models_key] = list(STACK_PRESETS[preset])

    def mark_custom():
        st.session_state[preset_key] = "Custom"

    st.subheader(label)
    st.selectbox(f"{label} preset", ["Custom", *STACK_PRESETS],
                 key=preset_key, on_change=apply_preset,
                 help="Selecting a preset replaces the list below. The all-models preset takes longer to train.")
    models = st.multiselect(
        f"{label} base models", STACK_BASE_OPTIONS, key=models_key, on_change=mark_custom,
        format_func=model_label,
        help="Add or remove models. Each stack needs at least two.",
    )
    combiner = st.selectbox(f"{label} combiner", META_MODELS, format_func=model_label)
    return models, combiner


st.set_page_config(page_title="Stroke Experiment Lab", layout="wide")


def require_password():
    expected = os.environ.get("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return
    st.title("Stroke Experiment Lab")
    entered = st.text_input("App password", type="password")
    if st.button("Open workspace", type="primary"):
        if hmac.compare_digest(entered, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


require_password()
st.title("Stroke Experiment Lab")
st.caption("Compare predictor sets for recorded stroke status · Research workspace")


@st.cache_resource(show_spinner="Loading saved experiments")
def restore_saved_runs():
    return sync_runs_from_database(RUNS)


if database_configured():
    try:
        restored_count = restore_saved_runs()
        if restored_count:
            st.toast(f"Restored {restored_count} saved experiment(s) from the database.")
    except Exception:
        st.warning("The database is temporarily unavailable; local results still work for this session.")

setup, results, comparison, plots = st.tabs([
    "New experiment", "Saved results", "Compare experiments", "Dataset plots",
])
with setup:
    controls, overview = st.columns([3, 2], gap="large")
    with controls:
        st.header("Choose predictors")
        st.write(f"The benchmark compares your selection with all {len(FEATURES)} predictors on the same development folds.")
        experiment_name = st.text_input(
            "Experiment name", max_chars=100, placeholder="Auto-generated from method, date and time (IST)"
        )
        selected = st.multiselect(
            "Input columns", list(FEATURES), default=DEMO_FEATURES, format_func=FEATURES.get
        )
        with st.expander("Feature-selection presets"):
            method = st.selectbox(
                "Method for the selected predictor set", PRESETS,
                help="Automatic methods select from your input columns using only training data.",
            )
            feature_count = st.number_input(
                "Features to keep", min_value=1, max_value=len(FEATURES), value=5,
                help="Applies to automatic methods. If you select fewer candidates, the run keeps them all.",
            )
    with overview:
        with st.container(border=True):
            st.subheader("Study dataset")
            st.markdown(f"[NHANES stroke dataset · Mendeley Data, version 1]({SOURCE})")
            if DATASET.exists():
                contents = DATASET.read_bytes()
                st.caption("Using the local study CSV.")
            else:
                uploaded = st.file_uploader("Upload the study CSV", type=["csv"])
                contents = uploaded.getvalue() if uploaded else None
            frame = None
            if contents:
                try:
                    frame, fingerprint = read_dataset(contents)
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
                else:
                    a, b = st.columns(2)
                    a.metric("Participants", f"{len(frame):,}")
                    b.metric("Stroke cases", f"{int(frame.stroke.sum()):,}")
            else:
                st.info("Download the CSV from the cited dataset and upload it here.")
    stack1, stack2 = st.columns(2, gap="large")
    with stack1:
        base_models, meta_model = stack_controls("Stack 1", DEFAULT_BASE_MODELS)
    with stack2:
        stack2_models, stack2_combiner = stack_controls("Stack 2", DEFAULT_STACK2)
    final_combiner = st.selectbox("Final combiner", META_MODELS, format_func=model_label,
                                 help="Combines Stack 1 and Stack 2 when you run both.")
    roles, _ = model_roles(base_models, meta_model, stack2_models, stack2_combiner, final_combiner)
    mode, tuning, invalid = tuning_controls(roles, model_label)
    enabled = mode != "Without tuning"
    single_roles, _ = model_roles(base_models, meta_model, None, stack2_combiner, final_combiner)

    def tuning_blocker(run_roles):
        labels = {role_label(role) for role in run_roles}
        if invalid & labels:
            return "Correct the highlighted parameter values in the model settings."
        if enabled and not any(profile["search"] for label, profile in tuning["profiles"].items()
                               if label in labels):
            return "Select a parameter to tune for a model in this run, or choose **Without tuning**."
        return ""

    both_blocker = tuning_blocker(roles)
    single_blocker = tuning_blocker(single_roles)
    if both_blocker and both_blocker == single_blocker:
        st.info(both_blocker)
    both_action, single_action = st.columns(2)
    use_two_stacks = both_action.button("Run both stacks", type="primary", disabled=bool(both_blocker))
    stack1_only = single_action.button("Run only Stack 1", disabled=bool(single_blocker))
    if both_blocker != single_blocker:
        if both_blocker:
            both_action.caption(both_blocker)
        if single_blocker:
            single_action.caption(single_blocker)
    if use_two_stacks or stack1_only:
        if not selected:
            st.error("Select at least one input column.")
        elif len(base_models) < 2:
            st.error("Select at least two Stack 1 base models.")
        elif use_two_stacks and len(stack2_models) < 2:
            st.error("Select at least two Stack 2 base models.")
        elif frame is None:
            st.error("Load the study CSV before running a comparison.")
        else:
            active_stack2 = stack2_models if use_two_stacks else None
            active_roles = roles if use_two_stacks else single_roles
            active_labels = {role_label(role) for role in active_roles}
            profiles = {label: profile for label, profile in tuning["profiles"].items() if label in active_labels}
            paired = mode == "With and without tuning"
            options = {**tuning, "profiles": profiles, "enabled": enabled}
            bar = None
            completed = []
            try:
                if invalid & active_labels:
                    raise ValueError("Correct the highlighted hyperparameter values before running.")
                validate_config(options, active_roles)
                bar = st.progress(0, text="Starting benchmark")
                pair_id = uuid4().hex if paired else None
                passes = [False, True] if paired else [enabled]
                for i, tune in enumerate(passes):
                    variant = "Tuned" if tune else "Untuned"

                    def progress(value, message):
                        bar.progress((i + value) / len(passes), text=f"{variant} · {message}")

                    name = experiment_name.strip()
                    if paired:
                        name = f"{(name or method)[:89]} · {variant}"
                    run_id = run_experiment(
                        frame, fingerprint, selected, method, progress,
                        base_models=base_models, meta_model=meta_model, experiment_name=name,
                        feature_count=feature_count, stack2_models=active_stack2,
                        stack2_combiner=stack2_combiner, final_combiner=final_combiner,
                        tuning={**options, "enabled": tune} if profiles else None, paired_id=pair_id,
                    )
                    completed.append(run_id)
                    st.session_state["result_id"] = run_id
                    st.session_state["saved_run"] = run_id
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                if bar is not None:
                    bar.empty()
                st.error(f"Training stopped. {exc}")
                if completed:
                    st.info("The untuned run finished and is available in Saved results.")
            else:
                bar.progress(1.0, text="Both runs saved" if paired else "Experiment saved")
                if paired:
                    st.session_state["compare-runs"] = completed
                    st.success("Both runs are ready. Open Compare experiments to see them side by side.")


def render_results(folder, section):
    widget_key = f"{section}-{folder.name}"
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    if settings.get("storage_warning"):
        st.warning(settings["storage_warning"])
    predictions = pd.read_csv(folder / "predictions.csv")
    metrics = saved_metrics(folder, predictions)
    features = pd.read_csv(folder / "features.csv")
    preprocessing = pd.read_csv(folder / "preprocessing.csv")
    st.caption(
        f"{settings['method']} · {settings['outer_folds']} development folds · "
        f"{settings['holdout_rows']:,} rows reserved for final testing"
    )
    st.text(experiment_label(settings))
    st.write("Selected inputs — " + ", ".join(FEATURES[name] for name in settings["selected"]))
    if "feature_count" in settings:
        st.caption(f"Selected subset: {settings['feature_count']} of {len(settings['selected'])} candidate predictors per fold.")
    saved_bases = settings.get("stack_base_models", DEFAULT_BASE_MODELS)
    saved_combiner = settings.get("stack_meta_model", META_MODELS[0])
    has_stack2 = "stack2_base_models" in settings
    st.write(f"**{'Stack 1' if has_stack2 else 'Stack'}:** {', '.join(saved_bases)} · **Combiner:** {saved_combiner}")
    if has_stack2:
        st.write(f"**Stack 2:** {', '.join(settings['stack2_base_models'])} · **Combiner:** {settings['stack2_combiner']}")
        st.write(f"**Final stack combiner:** {settings['final_combiner']}")
    counts = ["TN", "FP", "FN", "TP"]
    metric_names = [name for name in metrics if name not in {"Feature set", "Model", "Fold", *counts}]
    grouped = metrics.groupby(["Feature set", "Model"], sort=False)
    summary = grouped[metric_names].mean().join(grouped[counts].sum())
    summary = summary.sort_values("Average precision", ascending=False).reset_index()
    st.subheader("Model comparison")
    st.caption(
        f"Metrics are averaged across folds. Confusion counts are summed. "
        f"Classification metrics use a score cutoff of {settings['threshold']}. "
        f"{', '.join(name for name in metric_names if name in SCORE_METRICS)} use the model scores directly."
    )
    displayed_metrics = st.multiselect(
        "Metrics to display", metric_names + counts,
        default=[name for name in DEFAULT_METRICS if name in metric_names],
        key=f"{widget_key}-metrics",
    )
    sort_column = st.selectbox("Rank by", metric_names + counts, key=f"{widget_key}-rank",
                               help="Pick the metric you want to prioritize. Use Stroke recall to rank by the share of stroke cases caught.")
    lower = sort_column in {"Log loss", "Brier score", "Hamming loss", "FP", "FN"}
    order = st.radio("Sort order", ["Lowest first", "Highest first"] if lower else
                     ["Highest first", "Lowest first"], horizontal=True,
                     key=f"{widget_key}-order-{sort_column}")
    feature_filter = st.selectbox("Feature sets in table", ["Both", "All predictors", "Selected predictors"],
                                  key=f"{widget_key}-table-features")
    model_filter = st.multiselect("Filter models", sorted(summary.Model.unique()),
                                  key=f"{widget_key}-table-models", placeholder="All models")
    table = summary.sort_values(sort_column, ascending=order == "Lowest first", kind="stable")
    if feature_filter != "Both":
        table = table[table["Feature set"] == feature_filter]
    if model_filter:
        table = table[table.Model.isin(model_filter)]
    visible = list(dict.fromkeys(["Feature set", "Model", sort_column, *displayed_metrics]))
    st.dataframe(table[visible].round(4), hide_index=True)
    view = st.selectbox("Results view", [
        "Precision–recall curves", "ROC curves", "Confusion matrix", "Calibration", "Threshold trade-offs",
        "Neural training loss", "Selected features", "Preprocessing audit", "Fold metrics",
    ], key=f"{widget_key}-view")
    group = st.radio("Feature set", ["Selected predictors", "All predictors"], horizontal=True,
                     key=f"{widget_key}-features")
    subset = predictions[predictions["Feature set"] == group]
    if view in {"Precision–recall curves", "ROC curves", "Calibration"}:
        plot = go.Figure()
        for model, rows in subset.groupby("Model"):
            if view == "Precision–recall curves":
                y, x, _ = precision_recall_curve(rows.stroke, rows.score)
                axes = ("Recall", "Precision")
            elif view == "ROC curves":
                x, y, _ = roc_curve(rows.stroke, rows.score)
                axes = ("False-positive rate", "True-positive rate")
            else:
                y, x = calibration_curve(rows.stroke, rows.score, n_bins=8, strategy="quantile")
                axes = ("Mean predicted score", "Observed stroke fraction")
            plot.add_scatter(x=x, y=y, mode="lines+markers" if view == "Calibration" else "lines", name=model)
        if view == "Precision–recall curves":
            plot.add_hline(y=subset.stroke.mean(), line_dash="dash", annotation_text="Stroke prevalence")
        else:
            plot.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#64748b"})
        plot.update_layout(xaxis_title=axes[0], yaxis_title=axes[1], legend_title_text="Model")
        st.caption("Curves use pooled held-out predictions from the development folds.")
        st.plotly_chart(plot, key=f"{widget_key}-chart")
    elif view == "Confusion matrix":
        model = st.selectbox("Model", subset.Model.unique(), key=f"{widget_key}-confusion-model")
        counts = metrics[(metrics["Feature set"] == group) & (metrics.Model == model)][["TN", "FP", "FN", "TP"]].sum()
        labels = ["No stroke", "Stroke"]
        plot = px.imshow([[counts.TN, counts.FP], [counts.FN, counts.TP]], x=labels, y=labels,
                         text_auto=True, color_continuous_scale="Blues",
                         labels={"x": "Predicted", "y": "Actual", "color": "Count"})
        st.caption(f"Pooled development counts at a score cutoff of {settings['threshold']}.")
        st.plotly_chart(plot, key=f"{widget_key}-chart")
    elif view == "Threshold trade-offs":
        model = st.selectbox("Model", subset.Model.unique(), key=f"{widget_key}-threshold-model")
        rows = subset[subset.Model == model]
        precision, recall, thresholds = precision_recall_curve(rows.stroke, rows.score)
        plot = go.Figure()
        plot.add_scatter(x=thresholds, y=precision[:-1], name="Precision", mode="lines")
        plot.add_scatter(x=thresholds, y=recall[:-1], name="Stroke recall", mode="lines")
        plot.add_vline(x=settings["threshold"], line_dash="dash")
        plot.update_layout(xaxis_title="Score cutoff", yaxis_title="Development metric")
        st.plotly_chart(plot, key=f"{widget_key}-chart")
    elif view == "Neural training loss":
        loss = pd.read_csv(folder / "learning.csv")
        loss = loss[loss["Feature set"] == group].assign(Fold=lambda rows: rows.Fold.astype(str))
        plot = px.line(loss, x="Epoch", y="Loss", color="Model", line_dash="Fold",
                       labels={"Loss": "Weighted training loss"})
        st.plotly_chart(plot, key=f"{widget_key}-chart")
    elif view == "Selected features":
        display = features[features["Feature set"] == group].copy()
        display["Feature"] = display.Feature.map(FEATURES)
        count = display.Feature.value_counts().rename_axis("Feature").reset_index(name="Folds selected")
        st.plotly_chart(px.bar(count, x="Folds selected", y="Feature", orientation="h"),
                        key=f"{widget_key}-chart")
        st.dataframe(display, hide_index=True)
    elif view == "Preprocessing audit":
        st.write("Numerical inputs use median imputation, and category codes are one-hot encoded. Age and BMI are published as groups. Preprocessing is fitted inside each training fold.")
        st.dataframe(preprocessing[preprocessing["Feature set"] == group], hide_index=True)
    else:
        st.dataframe(metrics[metrics["Feature set"] == group].round(4), hide_index=True)
    render_trials(folder, settings, section)
    with st.expander("Export experiment"):
        st.download_button("Model comparison CSV", summary.to_csv(index=False), "model-comparison.csv",
                           key=f"{widget_key}-comparison-csv")
        files = ["metrics.csv", "predictions.csv", "features.csv", "preprocessing.csv",
                 "learning.csv", "splits.csv", "settings.json"]
        for file in files:
            data = metrics.to_csv(index=False) if file == "metrics.csv" else (folder / file).read_bytes()
            st.download_button(file, data, file_name=file, key=f"{widget_key}-{file}")


def experiment_label(settings):
    stamp = datetime.fromisoformat(settings["created_utc"]).astimezone(EXPERIMENT_TZ)
    date = f"{stamp:%Y-%m-%d %H:%M:%S}"
    name = settings.get("experiment_name")
    if name and not name.endswith(date):
        return f"{name} · {date} IST"
    return f"{name or settings['method'] + ' · ' + date} IST"


with setup:
    latest = st.session_state.get("result_id")
    if latest and (RUNS / latest / "settings.json").is_file():
        st.header("Latest experiment results")
        st.info("This run is also saved in the Saved results tab. Use that tab to review this run alongside earlier experiments.")
        render_results(RUNS / latest, "latest")


with results:
    st.header("Saved results")
    folders = sorted((path.parent for path in RUNS.glob("*/settings.json")), reverse=True)
    if not folders:
        st.info("Run a comparison to populate this workspace.")
    else:
        names = [path.name for path in folders]
        if st.session_state.get("saved_run") not in names:
            st.session_state["saved_run"] = names[0]
        labels = {}
        for folder in folders:
            settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
            labels[folder.name] = experiment_label(settings)
        chosen = st.selectbox("Experiment (India Standard Time)", names, key="saved_run", format_func=labels.get)
        render_results(RUNS / chosen, "saved")

with comparison:
    compare_runs(folders, experiment_label)

with plots:
    render_dataset_plots(frame)
