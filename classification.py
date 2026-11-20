from config import OUTPUT_DIR, RANDOM_STATE, show, finish_plot

# 1 Imports and load classification data


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)



df = pd.read_pickle(
    OUTPUT_DIR / "overlap_model_pages.pkl"
).copy()

print("Dataset shape:", df.shape)

print("\nPages by newspaper:")
print(df["newspaper"].value_counts())

print("\nUnique issues:", df["issue_id"].nunique())

print("\nYears:", sorted(df["year"].unique()))

assert df["filename"].is_unique
assert df["text_model"].notna().all()
assert df["issue_id"].notna().all()
assert set(df["newspaper"]) == {"IMVO", "ISIG"}

# 2 Examine the issue-level distribution


issue_df = (
    df.groupby("issue_id", as_index=False)
    .agg(
        newspaper=("newspaper", "first"),
        year=("year", "first"),
        n_pages=("filename", "count")
    )
)

# Confirm that every issue belongs to one newspaper and year.
assert (
    df.groupby("issue_id")["newspaper"].nunique().max()
    == 1
)

assert (
    df.groupby("issue_id")["year"].nunique().max()
    == 1
)

print("Total issues:", len(issue_df))

print("\nIssues by newspaper and year:")
show(
    pd.crosstab(
        issue_df["year"],
        issue_df["newspaper"]
    )
)

print("\nPages per issue:")
show(
    issue_df.groupby("newspaper")["n_pages"].describe()
)

# 3 Create a leakage-safe train/validation/test split


# Stratification label combines newspaper and year.
issue_df["stratum"] = (
    issue_df["newspaper"]
    + "_"
    + issue_df["year"].astype(str)
)

# First split: 60% training, 40% held out.
train_issues, heldout_issues = train_test_split(
    issue_df,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=issue_df["stratum"]
)

# Second split: held-out issues equally into validation/test.
val_issues, test_issues = train_test_split(
    heldout_issues,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=heldout_issues["stratum"]
)

train_ids = set(train_issues["issue_id"])
val_ids = set(val_issues["issue_id"])
test_ids = set(test_issues["issue_id"])

# No issue may occur in more than one split.
assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

assert train_ids | val_ids | test_ids == set(issue_df["issue_id"])

train_df = df[df["issue_id"].isin(train_ids)].copy()
val_df = df[df["issue_id"].isin(val_ids)].copy()
test_df = df[df["issue_id"].isin(test_ids)].copy()

print("TRAIN:", len(train_df), "pages,", len(train_ids), "issues")
print("VALIDATION:", len(val_df), "pages,", len(val_ids), "issues")
print("TEST:", len(test_df), "pages,", len(test_ids), "issues")

print("\nTraining distribution:")
show(pd.crosstab(train_df["year"], train_df["newspaper"]))

print("\nValidation distribution:")
show(pd.crosstab(val_df["year"], val_df["newspaper"]))

print("\nTest distribution:")
show(pd.crosstab(test_df["year"], test_df["newspaper"]))

# Save split membership for reproducibility.
split_membership = pd.concat([
    train_df[["filename", "issue_id"]].assign(split="train"),
    val_df[["filename", "issue_id"]].assign(split="validation"),
    test_df[["filename", "issue_id"]].assign(split="test")
])

split_membership.to_csv(
    OUTPUT_DIR / "classification_split_membership.csv",
    index=False
)

# 4 Prepare model inputs


X_train = train_df["text_model"]
X_val = val_df["text_model"]

y_train = train_df["newspaper"]
y_val = val_df["newspaper"]

LABELS = ["IMVO", "ISIG"]

print("Training pages:", len(X_train))
print("Validation pages:", len(X_val))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nValidation labels:")
print(y_val.value_counts())

assert X_train.index.is_unique
assert X_val.index.is_unique

# 5 Establish the majority-class baseline

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(
    np.zeros((len(y_train), 1)),
    y_train
)

baseline_predictions = baseline.predict(
    np.zeros((len(y_val), 1))
)

baseline_results = {
    "model": "Majority baseline",
    "accuracy": accuracy_score(
        y_val, baseline_predictions
    ),
    "balanced_accuracy": balanced_accuracy_score(
        y_val, baseline_predictions
    ),
    "macro_f1": f1_score(
        y_val,
        baseline_predictions,
        average="macro",
        zero_division=0
    )
}

print("Baseline validation performance:")
show(pd.DataFrame([baseline_results]))

print("\nClassification report:")
print(
    classification_report(
        y_val,
        baseline_predictions,
        labels=LABELS,
        zero_division=0
    )
)

# Train word-level and character-level TF-IDF classifiers

# 6: Define classification pipelines

word_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=r"(?u)\b\w\w+\b",
        sublinear_tf=True,
        max_df=0.95
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ))
])

char_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=(3, 5),
        sublinear_tf=True,
        max_df=0.95
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=RANDOM_STATE
    ))
])

print("Word-level and character-level pipelines created.")

# 7: Hyperparameter configurations

word_configs = [
    {
        "tfidf__ngram_range": ngram_range,
        "tfidf__min_df": min_df,
        "tfidf__max_features": 20000,
        "classifier__C": C,
        "classifier__class_weight": "balanced"
    }
    for ngram_range in [(1, 1), (1, 2)]
    for min_df in [2, 5]
    for C in [0.1, 1.0, 10.0]
]

char_configs = [
    {
        "tfidf__ngram_range": ngram_range,
        "tfidf__min_df": min_df,
        "tfidf__max_features": 30000,
        "classifier__C": C,
        "classifier__class_weight": "balanced"
    }
    for ngram_range in [(3, 5), (3, 6)]
    for min_df in [2, 5]
    for C in [0.1, 1.0, 10.0]
]

print("Word configurations:", len(word_configs))
print("Character configurations:", len(char_configs))
print("Total configurations:",
      len(word_configs) + len(char_configs))

# 8: Word-level model tuning

from sklearn.base import clone

def evaluate_configurations(
    base_pipeline,
    configurations,
    model_name
):
    results = []

    for i, params in enumerate(configurations, start=1):

        model = clone(base_pipeline)
        model.set_params(**params)

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        results.append({
            "model": model_name,
            "configuration": i,
            "ngram_range": str(params["tfidf__ngram_range"]),
            "min_df": params["tfidf__min_df"],
            "max_features": params["tfidf__max_features"],
            "C": params["classifier__C"],
            "accuracy": accuracy_score(
                y_val, predictions
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),
            "macro_f1": f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            )
        })

        print(
            f"{model_name} {i}/{len(configurations)} | "
            f"Macro-F1: {results[-1]['macro_f1']:.4f} | "
            f"Balanced accuracy: "
            f"{results[-1]['balanced_accuracy']:.4f}"
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            ["macro_f1", "balanced_accuracy"],
            ascending=False,
            kind="stable"
        )
        .reset_index(drop=True)
    )


word_results = evaluate_configurations(
    word_pipeline,
    word_configs,
    "Word TF-IDF + LR"
)

print("\nWord-level tuning results:")
show(word_results)

word_results.to_csv(
    OUTPUT_DIR / "word_model_validation_results.csv",
    index=False
)

# 9: Character-level model tuning

char_results = evaluate_configurations(
    char_pipeline,
    char_configs,
    "Character TF-IDF + LR"
)

print("\nCharacter-level tuning results:")
show(char_results)

char_results.to_csv(
    OUTPUT_DIR / "character_model_validation_results.csv",
    index=False
)

# 10: Compare the best configurations

best_word = word_results.iloc[0]
best_char = char_results.iloc[0]

comparison = pd.DataFrame([
    {
        "model": "Majority baseline",
        "accuracy": baseline_results["accuracy"],
        "balanced_accuracy": baseline_results["balanced_accuracy"],
        "macro_f1": baseline_results["macro_f1"]
    },
    {
        "model": "Word TF-IDF + LR",
        "accuracy": best_word["accuracy"],
        "balanced_accuracy": best_word["balanced_accuracy"],
        "macro_f1": best_word["macro_f1"]
    },
    {
        "model": "Character TF-IDF + LR",
        "accuracy": best_char["accuracy"],
        "balanced_accuracy": best_char["balanced_accuracy"],
        "macro_f1": best_char["macro_f1"]
    }
])

comparison = comparison.sort_values(
    "macro_f1",
    ascending=False
).reset_index(drop=True)

print("Validation model comparison:")
show(comparison.round(4))

print("\nBest word-level configuration:")
show(best_word.to_frame().T)

print("\nBest character-level configuration:")
show(best_char.to_frame().T)

comparison.to_csv(
    OUTPUT_DIR / "classification_validation_comparison.csv",
    index=False
)

ax = comparison.set_index("model")[
    ["balanced_accuracy", "macro_f1"]
].plot(
    kind="bar",
    figsize=(9, 5),
    rot=0
)

ax.set_title("Classification Performance on Validation Data")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)

plt.tight_layout()
finish_plot("validation_model_comparison")

# Final Test Evaluation

# 11: Fit final models using train + validation data

from sklearn.base import clone

trainval_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

X_trainval = trainval_df["text_model"]
y_trainval = trainval_df["newspaper"]

X_test = test_df["text_model"]
y_test = test_df["newspaper"]

# Recover the selected configurations.
best_word_idx = int(best_word["configuration"]) - 1
best_char_idx = int(best_char["configuration"]) - 1

best_word_params = word_configs[best_word_idx]
best_char_params = char_configs[best_char_idx]

final_word_model = clone(word_pipeline)
final_word_model.set_params(**best_word_params)

final_char_model = clone(char_pipeline)
final_char_model.set_params(**best_char_params)

# Ensure no issue-level overlap.
assert set(trainval_df["issue_id"]).isdisjoint(
    set(test_df["issue_id"])
)

print("Training + validation pages:", len(trainval_df))
print("Test pages:", len(test_df))

print("\nTraining final word model...")
final_word_model.fit(X_trainval, y_trainval)

print("Training final character model...")
final_char_model.fit(X_trainval, y_trainval)

print("\nBoth final models fitted.")

# 12: Evaluate on the untouched test set

test_predictions = {
    "Majority baseline": np.repeat(
        y_trainval.value_counts().idxmax(),
        len(y_test)
    ),
    "Word TF-IDF + LR": final_word_model.predict(X_test),
    "Character TF-IDF + LR": final_char_model.predict(X_test)
}

test_results = []

for name, predictions in test_predictions.items():

    test_results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "errors": int(np.sum(
            np.asarray(predictions) != y_test.to_numpy()
        ))
    })

test_results_df = pd.DataFrame(test_results)

print("FINAL HELD-OUT TEST RESULTS")
show(test_results_df.round(4))

test_results_df.to_csv(
    OUTPUT_DIR / "classification_final_test_results.csv",
    index=False
)

for name, predictions in test_predictions.items():
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        classification_report(
            y_test,
            predictions,
            labels=LABELS,
            zero_division=0
        )
    )

#  13: Confusion matrices

fig, axes = plt.subplots(
    1, 2,
    figsize=(11, 4.5)
)

for ax, (name, model) in zip(
    axes,
    [
        ("Word TF-IDF + LR", final_word_model),
        ("Character TF-IDF + LR", final_char_model)
    ]
):

    predictions = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=LABELS,
        ax=ax,
        cmap="Blues",
        colorbar=False
    )

    ax.set_title(name)

plt.tight_layout()
finish_plot("test_confusion_matrices")

# 14: Identify classification errors

error_df = test_df[
    ["filename", "newspaper", "year", "page",
     "issue_id", "model_word_count", "text_model"]
].copy()

error_df["word_prediction"] = (
    final_word_model.predict(X_test)
)

error_df["char_prediction"] = (
    final_char_model.predict(X_test)
)

error_df["word_error"] = (
    error_df["word_prediction"] != error_df["newspaper"]
)

error_df["char_error"] = (
    error_df["char_prediction"] != error_df["newspaper"]
)

print("Word model errors:", error_df["word_error"].sum())
print("Character model errors:", error_df["char_error"].sum())

print("\nCharacter-model misclassified pages:")

show(
    error_df.loc[
        error_df["char_error"],
        [
            "filename",
            "newspaper",
            "char_prediction",
            "year",
            "page",
            "model_word_count"
        ]
    ]
)

print("\nWord-model misclassified pages:")

show(
    error_df.loc[
        error_df["word_error"],
        [
            "filename",
            "newspaper",
            "word_prediction",
            "year",
            "page",
            "model_word_count"
        ]
    ]
)

error_df.to_csv(
    OUTPUT_DIR / "classification_test_page_predictions.csv",
    index=False
)

# 15: Most influential word features

word_vectorizer = final_word_model.named_steps["tfidf"]
word_classifier = final_word_model.named_steps["classifier"]

features = word_vectorizer.get_feature_names_out()
coefficients = word_classifier.coef_[0]

negative_class = word_classifier.classes_[0]
positive_class = word_classifier.classes_[1]

top_n = 25

positive_indices = np.argsort(coefficients)[-top_n:][::-1]
negative_indices = np.argsort(coefficients)[:top_n]

positive_features = pd.DataFrame({
    "feature": features[positive_indices],
    "coefficient": coefficients[positive_indices]
})

negative_features = pd.DataFrame({
    "feature": features[negative_indices],
    "coefficient": coefficients[negative_indices]
})

print(f"Features associated with {positive_class}:")
show(positive_features)

print(f"\nFeatures associated with {negative_class}:")
show(negative_features)

positive_features.to_csv(
    OUTPUT_DIR / "word_positive_features.csv",
    index=False
)

negative_features.to_csv(
    OUTPUT_DIR / "word_negative_features.csv",
    index=False
)

# Final Robustness Experiment.
# We have already examined the test results, so this must be reported as a post-hoc sensitivity analysis, not as a new independently validated primary result.

# 16: Targeted masking of publication-specific cues

import re

IDENTITY_ABLATION_PATTERNS = [
    # Full and abbreviated newspaper names
    r"\bimvo\b",
    r"\bmvo\b",
    r"\bsigidimi\b",
    r"\bisigidimi\b",
    r"\bzabantsundu\b",

    # Common publication-place and issue cues
    r"\blovedale\b",
    r"\bking\s+william'?s?\s+town\b",

    # Explicit years and numeric dates
    r"\b18\d{2}\b",

    # Frequently recurring publication/date terms
    r"\bngolwesi(?:[\w-]*)\b",
]

def mask_identity_cues(text):
    cleaned = str(text)

    for pattern in IDENTITY_ABLATION_PATTERNS:
        cleaned = re.sub(
            pattern,
            " ",
            cleaned,
            flags=re.IGNORECASE
        )

    return re.sub(r"\s+", " ", cleaned).strip()


df_masked = df.copy()

df_masked["text_masked"] = (
    df_masked["text_model"].apply(mask_identity_cues)
)

print("Original example:")
print(df_masked.iloc[0]["text_model"][:400])

print("\nMasked example:")
print(df_masked.iloc[0]["text_masked"][:400])

print("\nEmpty masked pages:",
      (df_masked["text_masked"].str.len() == 0).sum())

assert (
    df_masked["text_masked"].str.len() > 0
).all()

# 17: Refit both models on identity-masked text

masked_trainval = df_masked.loc[
    df_masked["issue_id"].isin(
        set(trainval_df["issue_id"])
    )
].copy()

masked_test = df_masked.loc[
    df_masked["issue_id"].isin(
        set(test_df["issue_id"])
    )
].copy()

masked_word_model = clone(word_pipeline)
masked_word_model.set_params(**best_word_params)

masked_char_model = clone(char_pipeline)
masked_char_model.set_params(**best_char_params)

masked_word_model.fit(
    masked_trainval["text_masked"],
    masked_trainval["newspaper"]
)

masked_char_model.fit(
    masked_trainval["text_masked"],
    masked_trainval["newspaper"]
)

masked_results = []

for name, model in [
    ("Word TF-IDF + LR", masked_word_model),
    ("Character TF-IDF + LR", masked_char_model)
]:

    predictions = model.predict(
        masked_test["text_masked"]
    )

    masked_results.append({
        "model": name,
        "accuracy": accuracy_score(
            masked_test["newspaper"],
            predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            masked_test["newspaper"],
            predictions
        ),
        "macro_f1": f1_score(
            masked_test["newspaper"],
            predictions,
            average="macro",
            zero_division=0
        ),
        "errors": int(
            np.sum(
                predictions
                != masked_test["newspaper"].to_numpy()
            )
        )
    })

masked_results_df = pd.DataFrame(masked_results)

print("Identity-masked test performance:")
show(masked_results_df.round(4))

masked_results_df.to_csv(
    OUTPUT_DIR / "identity_masked_test_results.csv",
    index=False
)

# 18: Evaluate the original models on non-front pages

nonfront_test = test_df.loc[
    test_df["page"] != 1
].copy()

nonfront_results = []

for name, model in [
    ("Word TF-IDF + LR", final_word_model),
    ("Character TF-IDF + LR", final_char_model)
]:

    predictions = model.predict(
        nonfront_test["text_model"]
    )

    nonfront_results.append({
        "model": name,
        "n_pages": len(nonfront_test),
        "accuracy": accuracy_score(
            nonfront_test["newspaper"],
            predictions
        ),
        "balanced_accuracy": balanced_accuracy_score(
            nonfront_test["newspaper"],
            predictions
        ),
        "macro_f1": f1_score(
            nonfront_test["newspaper"],
            predictions,
            average="macro",
            zero_division=0
        ),
        "errors": int(
            np.sum(
                predictions
                != nonfront_test["newspaper"].to_numpy()
            )
        )
    })

nonfront_results_df = pd.DataFrame(nonfront_results)

print("Non-front-page test performance:")
show(nonfront_results_df.round(4))

print("\nNon-front-page test distribution:")
print(nonfront_test["newspaper"].value_counts())

nonfront_results_df.to_csv(
    OUTPUT_DIR / "nonfront_test_results.csv",
    index=False
)

# 19: Compare original and robustness results

original_summary = (
    test_results_df[
        ["model", "accuracy", "balanced_accuracy",
         "macro_f1", "errors"]
    ]
    .query("model != 'Majority baseline'")
    .assign(condition="Original text")
)

masked_summary = (
    masked_results_df.assign(
        condition="Identity-masked text"
    )
)

nonfront_summary = (
    nonfront_results_df[
        ["model", "accuracy", "balanced_accuracy",
         "macro_f1", "errors"]
    ]
    .assign(condition="Non-front pages")
)

robustness_summary = pd.concat(
    [
        original_summary,
        masked_summary,
        nonfront_summary
    ],
    ignore_index=True
)

print("Robustness comparison:")
show(
    robustness_summary[
        [
            "condition",
            "model",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "errors"
        ]
    ].round(4)
)

robustness_summary.to_csv(
    OUTPUT_DIR / "classification_robustness_summary.csv",
    index=False
)

# 21: Shortcut-baseline figure (new)
# Compares the TF-IDF classifiers with trivial "shortcut" baselines that
# use only page length and the rate of English function words.

import re
from sklearn.tree import DecisionTreeClassifier

ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "to", "in", "for", "that", "is", "it", "with"
}

def english_rate(text):
    tokens = re.findall(r"[a-z]+", str(text).lower())
    if not tokens:
        return 0.0
    return sum(t in ENGLISH_FUNCTION_WORDS for t in tokens) / len(tokens)

def shortcut_features(frame, use_english):
    feats = pd.DataFrame({"word_count": frame["model_word_count"].to_numpy()})
    if use_english:
        feats["english_rate"] = frame["text_model"].apply(english_rate).to_numpy()
    return feats

shortcut_rows = []

for label, use_english in [
    ("Page length only (tree)", False),
    ("Length + English rate (tree)", True)
]:
    tree = DecisionTreeClassifier(max_depth=2, random_state=RANDOM_STATE)
    tree.fit(shortcut_features(trainval_df, use_english), y_trainval)
    preds = tree.predict(shortcut_features(test_df, use_english))

    shortcut_rows.append({
        "model": label,
        "group": "Shortcut baseline",
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0)
    })

def tfidf_row(label, group, results_df, model_name):
    value = results_df.loc[
        results_df["model"] == model_name, "macro_f1"
    ].iloc[0]
    return {"model": label, "group": group, "macro_f1": value}

figure_df = pd.DataFrame(
    [tfidf_row("Majority class", "Shortcut baseline",
               test_results_df, "Majority baseline")]
    + shortcut_rows
    + [
        tfidf_row("Word TF-IDF + LR", "TF-IDF classifier",
                  test_results_df, "Word TF-IDF + LR"),
        tfidf_row("Character TF-IDF + LR", "TF-IDF classifier",
                  test_results_df, "Character TF-IDF + LR"),
        tfidf_row("Word, identity-masked", "Post-hoc robustness",
                  masked_results_df, "Word TF-IDF + LR"),
        tfidf_row("Character, identity-masked", "Post-hoc robustness",
                  masked_results_df, "Character TF-IDF + LR"),
        tfidf_row("Word, non-front pages", "Post-hoc robustness",
                  nonfront_results_df, "Word TF-IDF + LR"),
        tfidf_row("Character, non-front pages", "Post-hoc robustness",
                  nonfront_results_df, "Character TF-IDF + LR"),
    ]
)

GROUP_COLOURS = {
    "Shortcut baseline": "#9a9a9a",
    "TF-IDF classifier": "#2a6fb0",
    "Post-hoc robustness": "#e08a2c",
}

fig, ax = plt.subplots(figsize=(8, 4.8))

plot_df = figure_df.iloc[::-1].reset_index(drop=True)
bars = ax.barh(
    plot_df["model"],
    plot_df["macro_f1"],
    color=[GROUP_COLOURS[g] for g in plot_df["group"]]
)

for bar, value in zip(bars, plot_df["macro_f1"]):
    ax.text(
        value + 0.01, bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}", va="center", fontsize=9
    )

ax.set_xlim(0, 1.12)
ax.set_xlabel("Macro-F1 on held-out test pages (224 pages, 44 issues)")
ax.set_title("Simple shortcuts vs TF-IDF classifiers")

handles = [
    plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOURS.values()
]
ax.legend(
    handles, GROUP_COLOURS.keys(),
    loc="upper center", bbox_to_anchor=(0.5, -0.16),
    ncol=3, fontsize=8, frameon=False
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "shortcut_baselines.png",
    dpi=200,
    bbox_inches="tight"
)
finish_plot()

figure_df.to_csv(OUTPUT_DIR / "shortcut_baselines.csv", index=False)
print(figure_df.round(4))

# 20: Save final models and experiment metadata

import json
import joblib
import sklearn

joblib.dump(
    final_word_model,
    OUTPUT_DIR / "final_word_classifier.joblib"
)

joblib.dump(
    final_char_model,
    OUTPUT_DIR / "final_character_classifier.joblib"
)

joblib.dump(
    masked_word_model,
    OUTPUT_DIR / "masked_word_classifier.joblib"
)

joblib.dump(
    masked_char_model,
    OUTPUT_DIR / "masked_character_classifier.joblib"
)

experiment_metadata = {
    "research_title": (
        "Language, Culture and Editorial Differences "
        "in Historical isiXhosa Newspapers"
    ),
    "random_state": RANDOM_STATE,
    "overlap_years": [1884, 1885, 1887, 1888],
    "classification_unit": "newspaper page",
    "split_unit": "newspaper issue",
    "primary_selection_metric": "validation macro-F1",
    "test_pages": int(len(test_df)),
    "test_issues": int(test_df["issue_id"].nunique()),
    "best_word_parameters": best_word_params,
    "best_character_parameters": best_char_params,
    "sklearn_version": sklearn.__version__,
    "robustness_note": (
        "Post-hoc sensitivity analyses; "
        "not independently held-out confirmatory results."
    )
}

with open(
    OUTPUT_DIR / "classification_experiment_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        experiment_metadata,
        f,
        indent=2,
        default=str
    )

print("Final models and experiment metadata saved.")