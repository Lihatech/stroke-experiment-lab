"""Interactive plots for development records."""

import pandas as pd
import plotly.express as px
import streamlit as st

from experiment import FEATURES, NUMERIC, development_split


def render_dataset_plots(frame):
    st.header("Dataset plots")
    if frame is None:
        st.info("Load the study CSV in New experiment.")
        return

    data, _ = development_split(frame)
    data = data.rename(columns=FEATURES).assign(
        **{"Stroke status": data.stroke.map({0: "No stroke", 1: "Stroke"})}
    )
    numerical = [label for name, label in FEATURES.items() if name in NUMERIC]
    categorical = [label for name, label in FEATURES.items() if name not in NUMERIC]
    st.caption("Plots use the development records. The final test records are reserved.")
    chart = st.selectbox("Plot type", [
        "Category bar chart", "Violin", "Box", "Histogram", "ECDF", "Strip",
        "Scatter", "Density heatmap", "Correlation heatmap", "Category heatmap",
        "Stroke class balance", "Missing values",
    ])
    grouping = chart not in {"Correlation heatmap", "Stroke class balance", "Missing values"}
    color = "Stroke status" if grouping and st.checkbox("Group by stroke status", value=True) else None
    if chart == "Category bar chart":
        names = st.multiselect("Categorical predictors", categorical,
                               default=["Hypertension", "Coronary heart disease", "Gender", "Smoking"])
        percent = st.radio("Bar values", ["Participant count", "Percent within category"],
                           horizontal=True) if color else "Participant count"
        if not names:
            st.info("Select at least one categorical predictor.")
        for name in names:
            dimensions = [name, color] if color else [name]
            counts = data.groupby(dimensions, dropna=False).size().reset_index(name="Participant count")
            counts[name] = counts[name].astype("string").fillna("Missing")
            counts["Percent within category"] = (
                100 * counts["Participant count"] / counts.groupby(name)["Participant count"].transform("sum")
            )
            if percent == "Percent within category" and color:
                figure = px.bar(counts, x=name, y=percent, color=color, title=name, barmode="stack")
            else:
                figure = px.bar(counts, x=name, y="Participant count", color=color, title=name, barmode="group")
            st.plotly_chart(figure, key=f"dataset-bar-{name}")
        return

    if chart in {"Violin", "Box", "Histogram", "ECDF", "Strip"}:
        names = st.multiselect("Numerical predictors", numerical, default=["Fasting Glucose"])
        if not names:
            st.info("Select at least one numerical predictor.")
        for name in names:
            if chart == "Violin":
                figure = px.violin(data, y=name, x=color, color=color, box=True, title=name)
            elif chart == "Box":
                figure = px.box(data, y=name, x=color, color=color, title=name)
            elif chart == "Histogram":
                figure = px.histogram(data, x=name, color=color, barmode="overlay", opacity=0.65, title=name)
            elif chart == "ECDF":
                figure = px.ecdf(data, x=name, color=color, title=name)
            else:
                figure = px.strip(data, y=name, x=color, color=color, title=name)
            st.plotly_chart(figure, key=f"dataset-distribution-{name}")
        return

    if chart in {"Scatter", "Density heatmap", "Category heatmap"}:
        options = categorical if chart == "Category heatmap" else numerical
        x = st.selectbox("X predictor", options)
        y = st.selectbox("Y predictor", [name for name in options if name != x])
        if chart == "Scatter":
            figure = px.scatter(data, x=x, y=y, color=color, opacity=0.55)
        elif chart == "Density heatmap":
            figure = px.density_heatmap(data, x=x, y=y, facet_col=color)
        else:
            panels = list(data.groupby(color)) if color else [("All development records", data)]
            for label, rows in panels:
                counts = pd.crosstab(rows[y], rows[x]).reindex(
                    index=sorted(data[y].dropna().unique()), columns=sorted(data[x].dropna().unique()),
                    fill_value=0,
                )
                counts.index, counts.columns = counts.index.astype(str), counts.columns.astype(str)
                figure = px.imshow(counts, text_auto=True, title=label,
                                   labels={"x": x, "y": y, "color": "Count"})
                st.plotly_chart(figure, key=f"dataset-category-{label}")
            return
    elif chart == "Correlation heatmap":
        names = st.multiselect("Numerical predictors", numerical, default=numerical[:6])
        method = st.selectbox("Correlation", ["Pearson", "Spearman"])
        if len(names) < 2:
            st.info("Select at least two predictors.")
            return
        figure = px.imshow(data[names].corr(method=method.lower()), zmin=-1, zmax=1,
                           color_continuous_scale="RdBu_r", text_auto=".2f")
    elif chart == "Stroke class balance":
        counts = data["Stroke status"].value_counts().reset_index(name="Participant count")
        figure = px.bar(counts, x="Stroke status", y="Participant count", text_auto=True)
    else:
        missing = data[list(FEATURES.values())].isna().sum().rename_axis("Predictor").reset_index(name="Missing values")
        figure = px.bar(missing, x="Missing values", y="Predictor", height=850)
    st.plotly_chart(figure, key="dataset-relationship")
