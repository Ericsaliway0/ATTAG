#!/usr/bin/env python3
# count_significance_in_interactions.py
# Reads a gene interaction results CSV and counts significant vs non-significant pairs (with percentages).

import pandas as pd

def main():
    file_path = "data/PathNet/gene_interaction_p_value_results_with_fdr_pathnet_two_commons.csv"
    file_path = "data/PPNet/gene_interaction_p_value_results_with_fdr_ppnet_two_commons.csv"

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Optional: print column names for verification
    print("Columns:", df.columns.tolist())

    # Count significance categories
    counts = df['significance'].value_counts()
    total = len(df)

    # Calculate percentages safely
    percent_significant = (counts.get('significant', 0) / total) * 100 if total > 0 else 0
    percent_nonsignificant = (counts.get('non-significant', 0) / total) * 100 if total > 0 else 0

    # Print results
    print("\nCounts:")
    print(counts)

    print(f"\n✅ Significant: {counts.get('significant', 0)} ({percent_significant:.2f}%)")
    print(f"❌ Non-significant: {counts.get('non-significant', 0)} ({percent_nonsignificant:.2f}%)")
    print(f"📊 Total: {total}")

if __name__ == "__main__":
    main()
