"""
evaluate_ragas.py

This script evaluates the retrieval performance of a job search pipeline using Recall@K.

1. Loads ground truth job links from `ground_truth.json`.
2. Loads predicted job links from `predictions.json`.
3. Merges them on the `query` field.
4. Computes Recall@K for K=1,3,5.
5. Outputs evaluation results.

Dependencies:
- Python 3.8+
- `pandas`, `json`
"""

import json
import pandas as pd

# 📂 File paths (Updated based on folder structure)
GROUND_TRUTH_FILE = "../Data_files/ground_truth.json"
PREDICTIONS_FILE = "../Data_files/prediction1.json"
EVALUATION_FILE = "../Data_files/evaluation_results.csv"

# ✅ Load ground truth data
try:
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)
    print(f"✅ Loaded {len(ground_truth_data)} ground truth queries.")
except Exception as e:
    raise ValueError(f"Error loading ground truth file: {e}")

# ✅ Load predictions data
try:
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        predictions_data = json.load(f)
    print(f"✅ Loaded {len(predictions_data)} predicted queries.")
except Exception as e:
    raise ValueError(f"Error loading predictions file: {e}")

# ✅ Convert ground truth & predictions into Pandas DataFrames
df_ground_truth = pd.DataFrame(ground_truth_data)
df_predictions = pd.DataFrame(predictions_data)

# ✅ Strip spaces from column names (avoids merge issues)
df_ground_truth.rename(columns=lambda x: x.strip(), inplace=True)
df_predictions.rename(columns=lambda x: x.strip(), inplace=True)

# ✅ Ensure expected columns exist
expected_columns = {"query", "relevant_jobs"}
if not expected_columns.issubset(df_ground_truth.columns):
    raise KeyError(f"Missing expected columns in ground truth: {df_ground_truth.columns.tolist()}")

if not expected_columns.issubset(df_predictions.columns):
    raise KeyError(f"Missing expected columns in predictions: {df_predictions.columns.tolist()}")

# ✅ Rename columns for consistency
df_ground_truth.rename(columns={"relevant_jobs": "ground_truth_links"}, inplace=True)
df_predictions.rename(columns={"relevant_jobs": "predicted_links"}, inplace=True)

# ✅ Merge both DataFrames on "query"
merged_df = df_ground_truth.merge(df_predictions, on="query", how="inner")

if merged_df.empty:
    raise ValueError("❌ No matching queries found in both files. Ensure queries are formatted identically.")

print(f"✅ Successfully merged {len(merged_df)} queries.")

# 🔢 Recall@K Calculation
def compute_recall_at_k(row, k):
    """
    Computes Recall@K for a given row.

    :param row: Pandas row containing ground truth and predicted job links.
    :param k: Number of top results to consider.
    :return: Recall@K score (float).
    """
    ground_truth_links = set(row["ground_truth_links"])  # True job links
    predicted_links = set(row["predicted_links"][:k])  # Top-k predicted job links

    if not ground_truth_links:
        return 0.0  # Avoid division by zero

    intersection = ground_truth_links.intersection(predicted_links)
    return len(intersection) / len(ground_truth_links)


# ✅ Compute Recall@K metrics
for k in [1, 3, 5]:
    merged_df[f"Recall@{k}"] = merged_df.apply(lambda row: compute_recall_at_k(row, k), axis=1)

# ✅ Display evaluation results
print("\n📊 Evaluation Results:")
print(merged_df[["query", "Recall@1", "Recall@3", "Recall@5"]].head(10))

# ✅ Save results to CSV for analysis
merged_df.to_csv(EVALUATION_FILE, index=False)
print(f"✅ Evaluation results saved to {EVALUATION_FILE}")
