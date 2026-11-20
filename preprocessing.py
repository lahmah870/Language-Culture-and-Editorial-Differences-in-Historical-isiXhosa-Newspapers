from config import OUTPUT_DIR, RANDOM_STATE, find_data_zip, show, finish_plot

# 1 Load and inspect the dataset


from pathlib import Path
from zipfile import ZipFile
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility

# Dataset location (searched for in data/)
ZIP_PATH = find_data_zip()

# Output directory

print("Dataset found:", ZIP_PATH.name)
print("Output directory:", OUTPUT_DIR.resolve())

# 2Extract text and metadata from the ZIP file


FILE_PATTERN = re.compile(
    r"^(IMVO|ISIG)_(\d{4}-\d{2}-\d{2})_p(\d+)\.txt$"
)

records = []
unmatched_files = []
replacement_char_files = []

with ZipFile(ZIP_PATH, "r") as archive:
    for member in sorted(archive.namelist()):

        if member.endswith("/") or member.startswith("__MACOSX/"):
            continue

        if not member.lower().endswith(".txt"):
            continue

        filename = Path(member).name
        match = FILE_PATTERN.fullmatch(filename)

        if match is None:
            unmatched_files.append(member)
            continue

        newspaper, date_string, page_string = match.groups()

        raw_bytes = archive.read(member)

        # Decode with replacement to preserve recoverable text.
        text = raw_bytes.decode("utf-8-sig", errors="replace")

        if "\ufffd" in text:
            replacement_char_files.append(filename)

        records.append({
            "filename": filename,
            "newspaper": newspaper,
            "date": pd.to_datetime(date_string),
            "page": int(page_string),
            "text_raw": text
        })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No valid newspaper files were loaded.")

df["year"] = df["date"].dt.year
df["issue_id"] = (
    df["newspaper"] + "_" +
    df["date"].dt.strftime("%Y-%m-%d")
)

df["char_count"] = df["text_raw"].str.len()
df["word_count"] = df["text_raw"].str.split().str.len()
df["line_count"] = df["text_raw"].apply(lambda x: len(str(x).splitlines()))

print("Dataset shape:", df.shape)
print("Unmatched filenames:", len(unmatched_files))
print("Files with decoding replacement characters:",
      len(replacement_char_files))

print("\nFirst five records:")
show(df[[
    "filename", "newspaper", "date",
    "page", "word_count"
]].head())

# 3 Data quality checks

print("DATASET QUALITY REPORT")
print("=" * 55)

print("\nNumber of pages:", len(df))
print("Number of newspaper issues:", df["issue_id"].nunique())

print("\nNewspaper distribution:")
print(df["newspaper"].value_counts())

print("\nMissing values:")
print(df.isna().sum())

empty_mask = df["text_raw"].str.strip().eq("")
print("\nEmpty text documents:", int(empty_mask.sum()))

print("Duplicate filenames:",
      int(df["filename"].duplicated().sum()))

print("Duplicate newspaper/date/page combinations:",
      int(df.duplicated(
          subset=["newspaper", "date", "page"]
      ).sum()))

print("Exact duplicate text documents:",
      int(df["text_raw"].duplicated().sum()))

print("\nDocument length statistics:")
show(
    df.groupby("newspaper")["word_count"]
      .describe()
      .round(2)
)

print("\nShortest documents:")
show(
    df.nsmallest(10, "word_count")[[
        "filename", "newspaper", "word_count"
    ]]
)

# 4. Explore historical content coverage by publication year
year_counts = pd.crosstab(
    df["year"],
    df["newspaper"]
)

print("Number of pages by publication year:")
show(year_counts)

# Identify years represented by both newspapers.
overlap_years = year_counts.index[
    (year_counts["IMVO"] > 0) &
    (year_counts["ISIG"] > 0)
].tolist()

print("\nOverlapping publication years:", overlap_years)

print("\nPages in overlapping years:")
show(year_counts.loc[overlap_years])

ax = year_counts.plot(
    kind="bar",
    figsize=(12, 5),
    width=0.8
)

ax.set_title("Historical newspaper pages by publication year")
ax.set_xlabel("Publication year")
ax.set_ylabel("Number of pages")
ax.legend(title="Newspaper")

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "pages_by_year.png",
    dpi=200,
    bbox_inches="tight"
)
finish_plot()

# 5 Document length distribution

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for newspaper in ["IMVO", "ISIG"]:
    subset = df.loc[
        df["newspaper"] == newspaper, "word_count"
    ]

    axes[0].hist(
        subset,
        bins=40,
        alpha=0.55,
        label=newspaper
    )

axes[0].set_title("Document length distribution")
axes[0].set_xlabel("Words per page")
axes[0].set_ylabel("Number of pages")
axes[0].legend()

data = [
    df.loc[df["newspaper"] == name, "word_count"]
    for name in ["IMVO", "ISIG"]
]

axes[1].boxplot(
    data,
    tick_labels=["IMVO", "ISIG"],
    showfliers=False
)

axes[1].set_title("Document lengths by newspaper")
axes[1].set_ylabel("Words per page")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "document_lengths.png",
    dpi=200,
    bbox_inches="tight"
)
finish_plot()

# 6 Inspect actual historical text

def show_samples(
    dataframe,
    newspaper,
    n=3,
    max_chars=1200,
    random_state=42
):
    subset = dataframe.loc[
        dataframe["newspaper"] == newspaper
    ]

    if subset.empty:
        print(f"No documents found for {newspaper}")
        return

    samples = subset.sample(
        n=min(n, len(subset)),
        random_state=random_state
    )

    for _, row in samples.iterrows():
        print("=" * 70)
        print("FILE:", row["filename"])
        print("DATE:", row["date"].date())
        print("WORDS:", row["word_count"])
        print("-" * 70)
        print(row["text_raw"][:max_chars])
        print()

print("IMVO ZABANTSUNDU SAMPLES")
show_samples(df, "IMVO")

print("\nISIGIDIMI SAMA-XOSA SAMPLES")
show_samples(df, "ISIG")

# 7 Save the exploratory data and analysis results


# Preserve all loaded records for subsequent stages.
dataset_path = OUTPUT_DIR / "historical_isixhosa_raw.pkl"
df.to_pickle(dataset_path)

# Save publication-year counts.
year_counts.to_csv(
    OUTPUT_DIR / "year_distribution.csv"
)

# Save document-level metadata without duplicating raw text.
metadata_columns = [
    "filename", "newspaper", "date", "year",
    "page", "issue_id", "char_count",
    "word_count", "line_count"
]

df[metadata_columns].to_csv(
    OUTPUT_DIR / "document_metadata.csv",
    index=False
)

# Basic assertions to catch accidental data loss.
assert len(df) == len(records)
assert df["filename"].is_unique
assert df["date"].notna().all()

print("Saved dataset:", dataset_path)
print("Saved metadata and publication statistics.")
print("Exploratory analysis complete.")

# 8 Inspect duplicate text groups


# Identify all rows belonging to exact-duplicate text groups.
duplicate_rows = df.loc[
    df.duplicated(subset="text_raw", keep=False)
].copy()

print("Rows in duplicate groups:", len(duplicate_rows))

show(
    duplicate_rows[
        ["filename", "newspaper", "date",
         "page", "word_count"]
    ].sort_values(["word_count", "filename"])
)

for text, group in duplicate_rows.groupby("text_raw"):
    print("=" * 60)
    print("Files:", group["filename"].tolist())
    print("Words:", len(text.split()))
    print("Preview:")
    print(repr(text[:400]))

# 8. Deduplicate and select overlapping years


# Preserve the original dataset.
df_analysis = df.copy()

# Record duplicates before removing them.
duplicate_mask = df_analysis.duplicated(
    subset="text_raw",
    keep="first"
)

removed_duplicates = df_analysis.loc[
    duplicate_mask,
    ["filename", "newspaper", "date", "page"]
].copy()

removed_duplicates.to_csv(
    OUTPUT_DIR / "removed_duplicates.csv",
    index=False
)

df_analysis = (
    df_analysis.loc[~duplicate_mask]
    .copy()
    .reset_index(drop=True)
)

# Determine overlap from the data, not hardcoded years.
year_table = pd.crosstab(
    df_analysis["year"],
    df_analysis["newspaper"]
)

overlap_years = year_table.index[
    (year_table["IMVO"] > 0) &
    (year_table["ISIG"] > 0)
].tolist()

df_overlap = (
    df_analysis.loc[
        df_analysis["year"].isin(overlap_years)
    ]
    .copy()
    .reset_index(drop=True)
)

print("Original pages:", len(df))
print("Removed duplicates:", len(removed_duplicates))
print("Deduplicated pages:", len(df_analysis))

print("\nOverlapping years:", overlap_years)
print("Pages in overlapping subset:", len(df_overlap))

print("\nDistribution:")
show(
    pd.crosstab(
        df_overlap["year"],
        df_overlap["newspaper"]
    )
)

print("\nNumber of issues per newspaper:")
print(
    df_overlap.groupby("newspaper")["issue_id"].nunique()
)

assert df_analysis["text_raw"].is_unique
assert df_overlap["year"].isin(overlap_years).all()

# 9. Conservative text normalisation and cleaning
import unicodedata

HEADER_PATTERNS = [
    re.compile(
        r"^\s*(?:\d+\s+)?IMVO\s+ZABANTSUNDU"
        r"\s*(?:\(\s*NATIVE\s+OPINION\s*\))?"
        r"\s*[.,;:\-–—]*",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"^\s*(?:\d+\s+)?ISIGIDIMI\s+SAMA[\s\-]*XOSA"
        r"\s*[.,;:\-–—]*",
        flags=re.IGNORECASE
    )
]

def clean_historical_text(text):
    """Conservative cleaning of historical newspaper text."""
    if not isinstance(text, str):
        raise TypeError("Expected text to be a string.")

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u00ad", "")

    # Remove an identifiable publication header at the start.
    for pattern in HEADER_PATTERNS:
        text = pattern.sub("", text, count=1)

    # Normalise whitespace without changing historical spelling.
    text = re.sub(r"\s+", " ", text).strip()

    return text


df_overlap["text_clean"] = (
    df_overlap["text_raw"].apply(clean_historical_text)
)

df_overlap["clean_word_count"] = (
    df_overlap["text_clean"].str.split().str.len()
)

print("Empty cleaned documents:",
      df_overlap["text_clean"].eq("").sum())

print("\nDocument lengths after cleaning:")
show(
    df_overlap.groupby("newspaper")["clean_word_count"]
    .describe()
    .round(2)
)

print("\nBefore cleaning:")
print(repr(df_overlap.iloc[0]["text_raw"][:350]))

print("\nAfter cleaning:")
print(repr(df_overlap.iloc[0]["text_clean"][:350]))

# 10 Examine short and unusually long pages
print("SHORT DOCUMENTS (fewer than 100 words)")

short_docs = df_overlap.loc[
    df_overlap["clean_word_count"] < 100
]

print("Count:", len(short_docs))

show(
    short_docs[
        ["filename", "newspaper", "clean_word_count"]
    ].head(20)
)

print("\nLONG DOCUMENTS (more than 4,000 words)")

long_docs = df_overlap.loc[
    df_overlap["clean_word_count"] > 4000
]

print("Count:", len(long_docs))

show(
    long_docs[
        ["filename", "newspaper", "clean_word_count"]
    ].head(20)
)

print("\nLength quantiles by newspaper:")
show(
    df_overlap.groupby("newspaper")["clean_word_count"]
    .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    .unstack()
    .round(0)
)

# 11 Save our research datasets


df_analysis.to_pickle(
    OUTPUT_DIR / "deduplicated_pages.pkl"
)

df_overlap.to_pickle(
    OUTPUT_DIR / "overlapping_years_clean.pkl"
)

print("Saved:")
print("- deduplicated_pages.pkl")
print("- overlapping_years_clean.pkl")
print("- removed_duplicates.csv")

print("\nFinal overlap dataset:")
print(df_overlap.shape)

# 12 Check the prepared dataset


# Continue from the existing notebook.
# df_overlap should have been created in Cell 8
# and cleaned in Cell 9.

required_columns = {
    "filename",
    "newspaper",
    "date",
    "year",
    "page",
    "issue_id",
    "text_raw",
    "text_clean",
    "clean_word_count"
}

missing_columns = required_columns - set(df_overlap.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}. "
        "Run Cells 8-9 first."
    )

print("Prepared dataset shape:", df_overlap.shape)

print("\nNewspaper distribution:")
print(df_overlap["newspaper"].value_counts())

print("\nOverlapping years:")
print(sorted(df_overlap["year"].unique()))

print("\nUnique issues:")
print(df_overlap.groupby("newspaper")["issue_id"].nunique())

assert df_overlap["text_raw"].is_unique
assert df_overlap["filename"].is_unique
assert df_overlap["text_clean"].notna().all()

# 13 Examine potential newspaper identity leakage
import re

identity_patterns = {
    "imvo_name": r"\bimvo\s+zabantsundu\b",
    "native_opinion": r"\bnative\s+opinion\b",
    "isigidimi_name": r"\bisigidimi\s+sama[\s-]*xosa\b",
}

identity_results = []

for newspaper, group in df_overlap.groupby("newspaper"):
    for pattern_name, pattern in identity_patterns.items():

        matches = group["text_clean"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )

        identity_results.append({
            "newspaper": newspaper,
            "pattern": pattern_name,
            "matching_pages": int(matches.sum()),
            "total_pages": len(group),
            "percentage": round(
                100 * matches.mean(), 2
            )
        })

identity_df = pd.DataFrame(identity_results)

show(identity_df)

identity_df.to_csv(
    OUTPUT_DIR / "identity_pattern_audit.csv",
    index=False
)

# 14 Examine the structure of newspaper pages


def paragraph_statistics(text):
    # Paragraphs separated by blank lines.
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text)
        if p.strip()
    ]

    lengths = [len(p.split()) for p in paragraphs]

    return pd.Series({
        "paragraph_count": len(paragraphs),
        "median_paragraph_words": (
            float(np.median(lengths)) if lengths else 0
        ),
        "max_paragraph_words": max(lengths, default=0)
    })


paragraph_stats = df_overlap["text_raw"].apply(
    paragraph_statistics
)

df_overlap = pd.concat(
    [
        df_overlap.reset_index(drop=True),
        paragraph_stats.reset_index(drop=True)
    ],
    axis=1
)

print("Paragraph statistics by newspaper:")

show(
    df_overlap.groupby("newspaper")[
        [
            "paragraph_count",
            "median_paragraph_words",
            "max_paragraph_words"
        ]
    ].describe().round(2)
)

print("\nExample paragraph boundaries:")

for newspaper in ["IMVO", "ISIG"]:
    sample = df_overlap.loc[
        df_overlap["newspaper"] == newspaper
    ].iloc[0]

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n+", sample["text_raw"]
        )
        if p.strip()
    ]

    print(f"\n{newspaper}: {sample['filename']}")
    print("Paragraph count:", len(paragraphs))

    for i, paragraph in enumerate(paragraphs[:5], 1):
        print(f"\nParagraph {i}:")
        print(paragraph[:350])

# 15 Remove explicit newspaper identity phrases


import re
import unicodedata

# Explicit newspaper names and title variants.
# These patterns are deliberately narrow.
IDENTITY_PATTERNS = [
    r"\bimvo\s+zabantsundu\b",
    r"\bnative\s+opinion\b",
    r"\bisigidimi\s+sama[\s-]*xosa\b",
]

compiled_patterns = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in IDENTITY_PATTERNS
]

def prepare_model_text(text):
    if not isinstance(text, str):
        raise TypeError("Expected a string.")

    text = unicodedata.normalize("NFC", text)

    # Remove explicit newspaper-identifying phrases.
    for pattern in compiled_patterns:
        text = pattern.sub(" ", text)

    # Normalise whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


df_overlap["text_model"] = (
    df_overlap["text_clean"].apply(prepare_model_text)
)

df_overlap["model_word_count"] = (
    df_overlap["text_model"].str.split().str.len()
)

print("Empty model documents:",
      df_overlap["text_model"].eq("").sum())

print("\nRemaining identifying expressions:")

for name, pattern in identity_patterns.items():
    count = df_overlap["text_model"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).sum()

    print(f"{name}: {count}")

print("\nModel text length statistics:")
show(
    df_overlap.groupby("newspaper")["model_word_count"]
    .describe()
    .round(2)
)

# 16 Create fixed-length text windows


WINDOW_SIZE = 400
MIN_WINDOW_WORDS = 200

def create_windows(text, window_size=400, min_words=200):
    words = text.split()
    windows = []

    for start in range(0, len(words), window_size):
        chunk = words[start:start + window_size]

        if len(chunk) >= min_words:
            windows.append(" ".join(chunk))

    return windows


window_records = []

for row in df_overlap.itertuples(index=False):

    windows = create_windows(
        row.text_model,
        window_size=WINDOW_SIZE,
        min_words=MIN_WINDOW_WORDS
    )

    for window_number, window_text in enumerate(windows):
        window_records.append({
            "filename": row.filename,
            "newspaper": row.newspaper,
            "date": row.date,
            "year": row.year,
            "issue_id": row.issue_id,
            "window_number": window_number,
            "text": window_text,
            "word_count": len(window_text.split())
        })

df_windows = pd.DataFrame(window_records)

if df_windows.empty:
    raise ValueError("No text windows were created.")

print("Total windows:", len(df_windows))

print("\nWindows by newspaper:")
print(df_windows["newspaper"].value_counts())

print("\nWindow length statistics:")
show(
    df_windows.groupby("newspaper")["word_count"]
    .describe()
    .round(2)
)

print("\nNumber of source pages represented:")
print(df_windows["filename"].nunique())

# 17 save Modelling data


df_overlap.to_pickle(
    OUTPUT_DIR / "overlap_model_pages.pkl"
)

df_windows.to_pickle(
    OUTPUT_DIR / "topic_windows.pkl"
)

# Record preprocessing configuration.
import json

configuration = {
    "research_question": (
        "How do linguistic and thematic patterns differ "
        "between Imvo Zabantsundu and Isigidimi sama-Xosa?"
    ),
    "overlap_years": [int(y) for y in overlap_years],
    "window_size": WINDOW_SIZE,
    "minimum_window_words": MIN_WINDOW_WORDS,
    "identity_patterns": IDENTITY_PATTERNS,
    "random_state": RANDOM_STATE
}

with open(
    OUTPUT_DIR / "preprocessing_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(configuration, file, indent=2)

print("Saved:")
print("- overlap_model_pages.pkl")
print("- topic_windows.pkl")
print("- preprocessing_config.json")