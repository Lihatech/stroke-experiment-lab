# Stroke Experiment Lab

A local browser app for comparing predictors and deep-learning models on Ping
Wang's [Imbalanced Data-based Prediction and Risk Factor Analysis of Stroke](https://data.mendeley.com/datasets/xggs239bnw/1)
(Mendeley Data v1, DOI 10.17632/xggs239bnw.1, CC BY 4.0).
The NHANES release contains 4,603 rows, 362 stroke cases, and 35 predictors.

## Run on Windows

Use a Python installation supported by the PyTorch Windows wheels.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Download `data.csv` from Mendeley Data. Upload it in the app or place it at
`data/nhanes-stroke-v1.csv`. The app checks the schema, expected class counts,
and SHA-256 fingerprint against the original version 1 CSV.
The original file must be used without saving it through a spreadsheet editor.
The release has no participant ID. The ID in run exports is the 1-based source
data row number, excluding the header.

Choose the predictors and run the benchmark. The results compare all eligible
predictors with the chosen subset on the same outer folds. The optional
feature-selection presets include Mutual Information, ANOVA, Pearson correlation,
Information Gain, L1 Logistic Regression, Particle Swarm Optimization (PSO), and
Harris Hawks Optimization (HHO). Set **Features to keep** to control the size of
the selected subset. Automatic methods fit separately in every training fold.

Information Gain uses up to five quantile bins for continuous measurements.
L1 Logistic Regression ranks each original feature by the largest absolute
coefficient among its encoded columns. PSO and HHO search fixed-size subsets
using eight agents and eight iterations. They score subsets by three-fold
average precision from a class-weighted logistic classifier fitted within
the selector's training data. These are fixed-size adaptations for NHANES.

Give the experiment a name, or leave it blank to use the feature selection
method and India Standard Time (IST). Completed results appear below the form
and are also available in the Saved results tab.

Choose a preset as a starting point and edit its base-model list directly.
Either stack can use any supported base model. The **All supported base models**
preset fills the entire list and takes longer to train. The MDPI presets match the
paper's reported base-model lineups and use the combiners you select. New runs
record the configuration and relevant package versions in `settings.json`.

## Evaluation protocol

Use **Dataset plots** to explore the development records. The categorical bar
charts open with hypertension, coronary heart disease, gender, and smoking.
Numerical distributions, scatter plots, and heatmaps use the predictors you choose.

Choose **Run both stacks** to train the final combiner, or **Run only Stack 1**
for a single-stack comparison. Selecting a preset fills the editable base-model
list. You can add or remove models before running.
Each stack has a preset or custom list and its own combiner. The final combiner
uses held-out predictions from Stack 1 and Stack 2. The second stack defaults
to the MDPI lineup: LightGBM, AdaBoost, XGBoost, and CatBoost. The broader
Boosting preset adds histogram gradient boosting and classic gradient boosting.
The combiner menus also include these boosters and classical classifiers such as
logistic regression, KNN, SVM, Extra Trees, and Gaussian Naive Bayes, plus a small MLP.
The additional training folds make two-stack runs slower.

## Hyperparameter tuning and run comparison

Open **Hyperparameter tuning and model settings** to choose a run mode and
configure models by role. Choose fixed baseline settings and the parameters
to tune. Trial values accept comma-separated lists and inclusive ranges such
as `0.01:0.1:5`. The advanced option shows additional scalar settings from
the installed estimator.

**Grid search** tries every combination for that model. **Random search**
samples up to the chosen budget. Set the scoring metric, search fold count,
and search seed. All fits use training-only preprocessing, feature selection,
and out-of-fold predictions. The final 20% remains reserved. More models,
search folds, or trials increase runtime, especially when tuning combiners.

**With and without tuning** saves two matched runs and preselects them in
**Compare experiments**. That tab also accepts older saved runs and compares
metrics, configuration, runtime, and charts. Differences from the reference
are shown when the dataset, validation splits, and cutoff match. Open
**Hyperparameter trial results** to review or export candidates and winners.

## Reading results

In **Model comparison**, use **Rank by**, **Sort order**, and the feature-set
and model filters to inspect the same metric across models. Each run evaluates
individual models, the equal-weight average of the distinct selected base
models, and the configured stacks on both feature sets.

- Stratified 80/20 split with seed 42. The development benchmark uses three
  stratified folds. The 20% holdout is reserved for a later final evaluation.
- The comparison includes individual models, the equal-weight average of the
  selected base models, and the configured stack. Within each outer training
  partition, three inner folds produce one held-out score per patient from
  each selected base model for the combiner.
- Imputation, encoding, scaling and any automatic feature selection are fitted
  on training rows. Published age and BMI group codes are treated as categorical.
  Individual benchmark models use class weighting during fit. The combiner
  learns from the original class distribution in the inner held-out scores.
- The leaderboard ranks by mean fold average precision. Threshold metrics use
  0.5. Curve views use pooled held-out development predictions. Weighted model
  scores can be poorly calibrated, so inspect the calibration plot before
  interpreting scores as probabilities.
- Use **Metrics to display** to choose columns. The full CSV includes average
  precision, ROC AUC, accuracy, stroke precision and recall, specificity,
  negative predictive value, F1, balanced accuracy, MCC, Cohen's kappa,
  Jaccard, Hamming loss, log loss, Brier score, and confusion counts.
- Run settings, split assignments, preprocessing summaries, selected features,
  predictions and metrics are saved under `runs/nhanes-v1`. Earlier Kaggle
  runs remain in their original folders.

The current milestone covers development comparisons. Explainability views
and the locked final evaluation are later milestones.

## Deploy on Render with Supabase

This app is prepared for a Render web service and a Supabase PostgreSQL
database. Render runs the long-lived Streamlit process; Supabase preserves
completed experiment folders across Render restarts. The local `runs` folder
is an ephemeral cache on the hosted service.

1. Push the project to a private, client-owned Git repository. Do not commit
   `.env`, the dataset, or `runs`.
2. Create a Supabase project. In **Connect**, copy the PostgreSQL **Session
   pooler** URI. The app creates its `stroke_experiments` table automatically.
3. In Render, choose **New > Blueprint**, connect the repository, and use the
   included `render.yaml`.
4. Set `DATABASE_URL` to the Supabase Session pooler URI and set a strong
   `APP_PASSWORD` in Render's environment settings. Never put either value in
   Git. SSL is enabled automatically by the app.
5. Deploy, open the Render URL, and upload the NHANES study CSV in the setup
   tab. The dataset is intentionally not stored in Git or PostgreSQL.

To copy current local experiments into Supabase once, set `DATABASE_URL` only
in the local terminal session and run:

```powershell
$env:DATABASE_URL = '<Supabase Session pooler URI>'
.\.venv\Scripts\python.exe migrate_runs.py
Remove-Item Env:DATABASE_URL
```

`APP_PASSWORD` is suitable for a single private client workspace. If several
independent users need accounts and private experiments, replace it later with
Supabase Auth and add an owner column plus row-level security.

To run the same default benchmark from PowerShell with the local CSV:

```powershell
.\.venv\Scripts\python.exe -c "from experiment import DATASET, DEMO_FEATURES, read_dataset, run_experiment; frame, digest = read_dataset(DATASET.read_bytes()); print(run_experiment(frame, digest, DEMO_FEATURES))"
```
