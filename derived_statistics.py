from config import OUTPUT_DIR

from pathlib import Path
import numpy as np
import pandas as pd

topic_summary = pd.read_csv(
    OUTPUT_DIR / "nmf_year_matched_topic_summary.csv", index_col=0
)

AD_TOPICS = ["Topic_3", "Topic_4", "Topic_5", "Topic_6", "Topic_8"]

ad_share = topic_summary[AD_TOPICS].sum(axis=1)
print("Advertisement share of topic mass:")
print(ad_share.round(4))

#Test issues per newspaper
pages = pd.read_pickle(OUTPUT_DIR / "overlap_model_pages.pkl")[
    ["filename", "newspaper", "issue_id"]
]
split_df = pd.read_csv(OUTPUT_DIR / "classification_split_membership.csv")

test_files = split_df.loc[split_df["split"] == "test", ["filename"]]
test_pages = test_files.merge(
    pages[["filename", "newspaper"]], on="filename", how="left"
)
test_issues = split_df.loc[
    split_df["split"] == "test", ["filename", "issue_id"]
].merge(pages[["filename", "newspaper"]], on="filename", how="left")

issues_per_paper = (
    test_issues.drop_duplicates("issue_id")["newspaper"].value_counts()
)
pages_per_paper = test_pages["newspaper"].value_counts()
pages_per_issue = (
    pages.groupby(["newspaper", "issue_id"]).size()
    .groupby("newspaper").mean()
)

print("\nTest pages by newspaper:")
print(pages_per_paper)
print("\nTest issues by newspaper:")
print(issues_per_paper, "| total:", int(issues_per_paper.sum()))
print("\nMean pages per issue (whole corpus):")
print(pages_per_issue.round(2))

# Validation errors of the selected models
n_val = int((split_df["split"] == "validation").sum())

word_results = pd.read_csv(OUTPUT_DIR / "word_model_validation_results.csv")
char_results = pd.read_csv(OUTPUT_DIR / "character_model_validation_results.csv")

rows = []
for name, results in [("Word TF-IDF + LR", word_results),
                      ("Character TF-IDF + LR", char_results)]:
    best = results.iloc[0]
    errors = int(round((1 - best["accuracy"]) * n_val))

    # Ties
    tied = results[np.isclose(results["macro_f1"], best["macro_f1"])]

    # Grid edge
    rows.append({
        "model": name,
        "validation_pages": n_val,
        "validation_errors": errors,
        "validation_macro_f1": round(best["macro_f1"], 4),
        "best_C": best["C"],
        "largest_C_tried": results["C"].max(),
        "best_C_at_grid_edge": bool(best["C"] == results["C"].max()),
        "configs_tied_for_best": len(tied),
        "tied_configuration_ids": tied["configuration"].tolist(),
    })

validation_summary = pd.DataFrame(rows)
print("\nValidation errors, ties and grid edge:")
print(validation_summary.T)

# save everything
derived = {
    "ad_share_IMVO": float(ad_share["IMVO"]),
    "ad_share_ISIG": float(ad_share["ISIG"]),
    "test_issues_IMVO": int(issues_per_paper.get("IMVO", 0)),
    "test_issues_ISIG": int(issues_per_paper.get("ISIG", 0)),
    "test_issues_total": int(issues_per_paper.sum()),
}
pd.Series(derived).to_csv(OUTPUT_DIR / "derived_statistics.csv", header=["value"])
validation_summary.to_csv(
    OUTPUT_DIR / "validation_selection_summary.csv", index=False
)