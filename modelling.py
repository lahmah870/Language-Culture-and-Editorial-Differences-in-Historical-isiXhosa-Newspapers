from config import OUTPUT_DIR, RANDOM_STATE, show, finish_plot

# 1. Load prepared data


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF



df_pages = pd.read_pickle(
    OUTPUT_DIR / "overlap_model_pages.pkl"
)

df_windows = pd.read_pickle(
    OUTPUT_DIR / "topic_windows.pkl"
)

print("Pages:", len(df_pages))
print("Windows:", len(df_windows))

print("\nPages by newspaper:")
print(df_pages["newspaper"].value_counts())

print("\nWindows by newspaper:")
print(df_windows["newspaper"].value_counts())

assert df_pages["filename"].is_unique
assert df_windows["text"].notna().all()
assert df_windows["issue_id"].notna().all()

# 2. Identify pages excluded by segmentation


represented_files = set(df_windows["filename"])

excluded_pages = df_pages.loc[
    ~df_pages["filename"].isin(represented_files),
    ["filename", "newspaper", "year", "model_word_count"]
].copy()

print("Pages without retained windows:", len(excluded_pages))

show(
    excluded_pages.sort_values("model_word_count")
)

print("\nExcluded pages by newspaper:")
print(excluded_pages["newspaper"].value_counts())

excluded_pages.to_csv(
    OUTPUT_DIR / "topic_excluded_pages.csv",
    index=False
)

# 3. Create a balanced topic-discovery sample (Sample one window per page, then balance the number of pages contributed by each newspaper.)


# Randomly select one window per source page.
# Fixed seed ensures reproducibility.

one_window_per_page = (
    df_windows
    .sample(frac=1, random_state=RANDOM_STATE)
    .drop_duplicates(subset="filename")
    .copy()
)

# Determine the smaller newspaper sample size.
sample_size = (
    one_window_per_page["newspaper"]
    .value_counts()
    .min()
)

# Sample the same number of pages from each newspaper.
balanced_parts = []

for newspaper in ["IMVO", "ISIG"]:
    subset = one_window_per_page.loc[
        one_window_per_page["newspaper"] == newspaper
    ]

    sampled = subset.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    balanced_parts.append(sampled)

df_topic_train = (
    pd.concat(balanced_parts, ignore_index=True)
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print("Balanced training windows:", len(df_topic_train))

print("\nNewspaper distribution:")
print(df_topic_train["newspaper"].value_counts())

print("\nYear distribution:")
show(
    pd.crosstab(
        df_topic_train["year"],
        df_topic_train["newspaper"]
    )
)

assert df_topic_train["filename"].is_unique
assert (
    df_topic_train["newspaper"].value_counts().nunique() == 1
)

# 4. Create a TF-IDF matrix


vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents=None,
    token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    min_df=5,
    max_df=0.85,
    max_features=12000,
    sublinear_tf=True,
    norm="l2"
)

X_train = vectorizer.fit_transform(
    df_topic_train["text"]
)

feature_names = vectorizer.get_feature_names_out()

print("TF-IDF matrix shape:", X_train.shape)
print("Vocabulary size:", len(feature_names))
print("Non-zero values:", X_train.nnz)

print("\nExample vocabulary terms:")
print(feature_names[:30])

# 5 Fit NMF models with different topic counts


TOPIC_COUNTS = [4, 6, 8, 10]

nmf_models = {}
topic_results = []

for n_topics in TOPIC_COUNTS:

    print(f"\nTraining NMF with {n_topics} topics...")

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=400,
        tol=1e-4
    )

    W = model.fit_transform(X_train)

    nmf_models[n_topics] = model

    topic_results.append({
        "n_topics": n_topics,
        "reconstruction_error": model.reconstruction_err_,
        "iterations": model.n_iter_,
        "converged": model.n_iter_ < model.max_iter
    })

results_df = pd.DataFrame(topic_results)

print("\nNMF model comparison:")
show(results_df)

results_df.to_csv(
    OUTPUT_DIR / "nmf_model_comparison.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["n_topics"],
    results_df["reconstruction_error"],
    marker="o"
)

plt.xlabel("Number of topics")
plt.ylabel("Reconstruction error")
plt.title("NMF Reconstruction Error by Topic Count")
plt.xticks(TOPIC_COUNTS)
plt.grid(alpha=0.3)
plt.tight_layout()
finish_plot("nmf_reconstruction_error")

# 6 Examine the discovered topics


def get_top_topic_words(model, feature_names, top_n=15):
    topic_rows = []

    for topic_idx, weights in enumerate(model.components_):

        top_indices = np.argsort(weights)[::-1][:top_n]

        top_words = [
            feature_names[i]
            for i in top_indices
        ]

        topic_rows.append({
            "topic": topic_idx + 1,
            "top_words": ", ".join(top_words)
        })

    return pd.DataFrame(topic_rows)


for n_topics in TOPIC_COUNTS:

    print("\n" + "=" * 75)
    print(f"NMF MODEL: {n_topics} TOPICS")
    print("=" * 75)

    topic_table = get_top_topic_words(
        nmf_models[n_topics],
        feature_names,
        top_n=15
    )

    show(topic_table)

    topic_table.to_csv(
        OUTPUT_DIR / f"nmf_topics_{n_topics}.csv",
        index=False
    )

# 7 Display complete topic vocabulary (Print the complete top 20 terms for every topic in the 8-topic model.)

# Use the already fitted 8-topic model.
model_8 = nmf_models[8]

TOP_N = 20

for topic_idx, weights in enumerate(model_8.components_):
    top_indices = np.argsort(weights)[::-1][:TOP_N]

    top_words = [
        feature_names[i]
        for i in top_indices
    ]

    print("=" * 75)
    print(f"TOPIC {topic_idx + 1}")
    print("=" * 75)
    print(", ".join(top_words))
    print()

# 8 Inspect representative passages for each topic


# Obtain topic weights for the balanced training corpus.
W_8 = model_8.transform(X_train)

TOP_EXAMPLES = 3
PREVIEW_CHARS = 1200

for topic_idx in range(model_8.n_components):

    print("\n" + "=" * 85)
    print(f"TOPIC {topic_idx + 1}")
    print("=" * 85)

    top_indices = np.argsort(
        W_8[:, topic_idx]
    )[::-1][:TOP_EXAMPLES]

    for rank, idx in enumerate(top_indices, start=1):

        row = df_topic_train.iloc[idx]

        print(f"\nExample {rank}")
        print("Newspaper:", row["newspaper"])
        print("Year:", row["year"])
        print("File:", row["filename"])
        print("Topic weight:", round(W_8[idx, topic_idx], 4))
        print("-" * 70)
        print(row["text"][:PREVIEW_CHARS])
        print()

# 9 Check whether topics are concentrated in one newspaper


# Normalise topic weights within each document.
row_sums = W_8.sum(axis=1, keepdims=True)

W_normalised = np.divide(
    W_8,
    row_sums,
    out=np.zeros_like(W_8),
    where=row_sums > 0
)

topic_columns = [
    f"Topic_{i+1}"
    for i in range(model_8.n_components)
]

topic_weights_df = pd.DataFrame(
    W_normalised,
    columns=topic_columns
)

topic_weights_df["newspaper"] = (
    df_topic_train["newspaper"].to_numpy()
)

topic_weights_df["year"] = (
    df_topic_train["year"].to_numpy()
)

topic_summary = (
    topic_weights_df
    .groupby("newspaper")[topic_columns]
    .mean()
)

print("Mean relative topic weights by newspaper:")
show(topic_summary.round(4))

# Plot comparison.
ax = topic_summary.T.plot(
    kind="bar",
    figsize=(11, 5),
    width=0.8
)

ax.set_title(
    "NMF Topic Weights by Newspaper "
    "(Balanced Training Sample)"
)
ax.set_xlabel("Topic")
ax.set_ylabel("Mean relative topic weight")
ax.legend(title="Newspaper")

plt.xticks(rotation=0)
plt.tight_layout()
finish_plot("nmf_topic_weights_balanced_sample")

topic_summary.to_csv(
    OUTPUT_DIR / "initial_topic_weights_by_newspaper.csv"
)

# 10 Examine year-specific topic patterns


year_topic_summary = (
    topic_weights_df
    .groupby(["year", "newspaper"])[topic_columns]
    .mean()
)

print("Mean topic weights by year and newspaper:")

show(
    year_topic_summary.round(3)
)

year_topic_summary.to_csv(
    OUTPUT_DIR / "initial_topic_weights_by_year.csv"
)

print("\nTraining windows per year and newspaper:")

show(
    pd.crosstab(
        topic_weights_df["year"],
        topic_weights_df["newspaper"]
    )
)

# Test Whether Our Topic Differences Are Robust

# 11 Calculate page-level topic weights


# Apply the existing 8-topic model to ALL available windows.
# Do not refit the vectorizer or NMF model.

X_all = vectorizer.transform(df_windows["text"])
W_all = model_8.transform(X_all)

# Convert to relative weights per window.
row_totals = W_all.sum(axis=1, keepdims=True)

W_all_relative = np.divide(
    W_all,
    row_totals,
    out=np.zeros_like(W_all),
    where=row_totals > 0
)

topic_columns = [
    f"Topic_{i + 1}"
    for i in range(model_8.n_components)
]

all_window_topics = df_windows[
    ["filename", "newspaper", "year", "issue_id"]
].reset_index(drop=True).copy()

for i, column in enumerate(topic_columns):
    all_window_topics[column] = W_all_relative[:, i]

# Each source page receives equal weight.
page_topics = (
    all_window_topics
    .groupby(
        ["filename", "newspaper", "year", "issue_id"],
        as_index=False
    )[topic_columns]
    .mean()
)

print("Pages with topic weights:", len(page_topics))

page_summary = (
    page_topics.groupby("newspaper")[topic_columns]
    .mean()
)

print("\nMean topic weights by newspaper (page-level):")
show(page_summary.round(4))

page_topics.to_csv(
    OUTPUT_DIR / "nmf_page_topic_weights.csv",
    index=False
)

page_summary.to_csv(
    OUTPUT_DIR / "nmf_page_topic_summary.csv"
)

assert page_topics["filename"].is_unique
assert len(page_topics) == df_windows["filename"].nunique()

# 12 year matched topic comparison


matched_parts = []
match_counts = []

for year in sorted(page_topics["year"].unique()):

    year_data = page_topics.loc[
        page_topics["year"] == year
    ]

    counts = year_data["newspaper"].value_counts()

    if not {"IMVO", "ISIG"}.issubset(counts.index):
        continue

    n_per_newspaper = int(
        min(counts["IMVO"], counts["ISIG"])
    )

    for newspaper in ["IMVO", "ISIG"]:

        subset = year_data.loc[
            year_data["newspaper"] == newspaper
        ]

        selected = subset.sample(
            n=n_per_newspaper,
            random_state=RANDOM_STATE
        )

        matched_parts.append(selected)

    match_counts.append({
        "year": year,
        "pages_per_newspaper": n_per_newspaper
    })

matched_pages = pd.concat(
    matched_parts,
    ignore_index=True
)

print("Year-matched pages:", len(matched_pages))

print("\nMatched distribution:")
show(
    pd.crosstab(
        matched_pages["year"],
        matched_pages["newspaper"]
    )
)

matched_summary = (
    matched_pages
    .groupby("newspaper")[topic_columns]
    .mean()
)

print("\nYear-matched topic comparison:")
show(matched_summary.round(4))

print("\nImvo minus Isigidimi:")
topic_differences = (
    matched_summary.loc["IMVO"]
    - matched_summary.loc["ISIG"]
)

show(
    topic_differences.to_frame(
        name="IMVO_minus_ISIG"
    ).round(4)
)

matched_pages.to_csv(
    OUTPUT_DIR / "nmf_year_matched_pages.csv",
    index=False
)

matched_summary.to_csv(
    OUTPUT_DIR / "nmf_year_matched_topic_summary.csv"
)

# 13 Visualise the year-matched comparison


ax = matched_summary.T.plot(
    kind="bar",
    figsize=(11, 5),
    width=0.8
)

ax.set_title(
    "NMF Topic Weights by Newspaper "
    "(Year-Matched Pages)"
)

ax.set_xlabel("NMF Topic")
ax.set_ylabel("Mean Relative Topic Weight")
ax.legend(title="Newspaper")

plt.xticks(rotation=0)
plt.tight_layout()
finish_plot("nmf_topic_weights_year_matched")

# Controlled Topic-Modelling Refinement

# 14 Check page-number distributions


# Add original page numbers to the page-level topic dataset.
page_metadata = df_pages[
    ["filename", "page"]
].drop_duplicates("filename")

page_audit = page_topics.merge(
    page_metadata,
    on="filename",
    how="left",
    validate="one_to_one"
)

print("Missing page numbers:", page_audit["page"].isna().sum())

print("\nPages by newspaper and page number:")
show(
    pd.crosstab(
        page_audit["page"],
        page_audit["newspaper"]
    )
)

print("\nNon-front pages by year:")
show(
    pd.crosstab(
        page_audit.loc[page_audit["page"] != 1, "year"],
        page_audit.loc[page_audit["page"] != 1, "newspaper"]
    )
)

assert page_audit["page"].notna().all()

# 15 Create a year-matched, non-front-page sample


# Restrict to pages other than page 1.
nonfront_pages = df_pages.loc[
    df_pages["page"] != 1
].copy()

# Keep only pages that have at least one retained window.
nonfront_pages = nonfront_pages.loc[
    nonfront_pages["filename"].isin(
        df_windows["filename"]
    )
].copy()

# Choose one random window per page.
nonfront_windows = df_windows.loc[
    df_windows["filename"].isin(
        nonfront_pages["filename"]
    )
]

one_per_page = (
    nonfront_windows
    .sample(frac=1, random_state=RANDOM_STATE)
    .drop_duplicates("filename")
    .copy()
)

# Match newspaper sample sizes within each year.
matched_parts = []

for year in sorted(one_per_page["year"].unique()):

    year_data = one_per_page.loc[
        one_per_page["year"] == year
    ]

    counts = year_data["newspaper"].value_counts()

    if not {"IMVO", "ISIG"}.issubset(counts.index):
        continue

    n = int(min(counts["IMVO"], counts["ISIG"]))

    for newspaper in ["IMVO", "ISIG"]:
        subset = year_data.loc[
            year_data["newspaper"] == newspaper
        ]

        matched_parts.append(
            subset.sample(
                n=n,
                random_state=RANDOM_STATE
            )
        )

df_refined_train = (
    pd.concat(matched_parts, ignore_index=True)
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print("Refined training windows:", len(df_refined_train))

print("\nYear-matched distribution:")
show(
    pd.crosstab(
        df_refined_train["year"],
        df_refined_train["newspaper"]
    )
)

print("\nUnique source pages:",
      df_refined_train["filename"].nunique())

assert df_refined_train["filename"].is_unique

# 16 Fit a refined NMF model


refined_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents=None,
    token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    min_df=5,
    max_df=0.85,
    max_features=12000,
    sublinear_tf=True,
    norm="l2"
)

X_refined = refined_vectorizer.fit_transform(
    df_refined_train["text"]
)

refined_model = NMF(
    n_components=8,
    init="nndsvda",
    random_state=RANDOM_STATE,
    max_iter=400,
    tol=1e-4
)

W_refined = refined_model.fit_transform(X_refined)

refined_features = (
    refined_vectorizer.get_feature_names_out()
)

print("Refined TF-IDF shape:", X_refined.shape)
print("Reconstruction error:", refined_model.reconstruction_err_)
print("Iterations:", refined_model.n_iter_)
print("Converged:", refined_model.n_iter_ < refined_model.max_iter)

print("\nRefined topic vocabulary:")

for topic_idx, weights in enumerate(refined_model.components_):

    top_indices = np.argsort(weights)[::-1][:20]

    top_words = [
        refined_features[i]
        for i in top_indices
    ]

    print(f"\nTOPIC {topic_idx + 1}")
    print(", ".join(top_words))

# 17 Inspect representative passages from the refined model


for topic_idx in range(refined_model.n_components):

    print("\n" + "=" * 80)
    print(f"REFINED TOPIC {topic_idx + 1}")
    print("=" * 80)

    top_indices = np.argsort(
        W_refined[:, topic_idx]
    )[::-1][:3]

    for rank, idx in enumerate(top_indices, start=1):

        row = df_refined_train.iloc[idx]

        print(f"\nExample {rank}")
        print("Newspaper:", row["newspaper"])
        print("Year:", row["year"])
        print("File:", row["filename"])
        print("Topic weight:",
              round(W_refined[idx, topic_idx], 4))
        print("-" * 60)
        print(row["text"][:1000])