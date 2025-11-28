"""
Run this BEFORE running mpi_bootstrap.py

Usage:
  python prepare_data.py
"""

import pandas as pd
import numpy as np

# ============================================================
# Configuration
# ============================================================
# Update this path to match your data location
RAW_DATA_FILE = "U.S._Chronic_Disease_Indicators.csv"
OUTPUT_FILE = "cdi_model_data.csv"

# ============================================================
# Main Script
# ============================================================

def main():
    print("=" * 50)
    print("Data Preparation for MPI Bootstrap Analysis")
    print("=" * 50)

    # ----------------------------------------------------------
    # Step 1: Load raw data
    # ----------------------------------------------------------
    print(f"\nLoading: {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)
    print(f"Raw data shape: {df.shape}")

    # ----------------------------------------------------------
    # Step 2: Basic cleaning
    # ----------------------------------------------------------
    # Convert DataValue to numeric
    df["DataValue"] = pd.to_numeric(df["DataValue"], errors="coerce")

    # Keep valid percentage range (0-100)
    df = df[(df["DataValue"] >= 0) & (df["DataValue"] <= 100)]
    print(f"After percentage filter: {len(df)} rows")

    # ----------------------------------------------------------
    # Step 3: Keep only U.S. states (exclude territories)
    # ----------------------------------------------------------
    valid_states = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
    ]
    df = df[df["LocationAbbr"].isin(valid_states)]
    print(f"After state filter: {len(df)} rows")

    # ----------------------------------------------------------
    # Step 4: Keep only percentage-based indicators
    # ----------------------------------------------------------
    valid_types = [
        "Crude Prevalence", "Age-adjusted Prevalence",
        "Crude Percentage", "Age-adjusted Percentage"
    ]
    df = df[df["DataValueType"].isin(valid_types)]
    print(f"After indicator type filter: {len(df)} rows")

    # ----------------------------------------------------------
    # Step 5: Select RQ1 indicators and pivot to wide format
    # ----------------------------------------------------------
    selected_questions = [
        "Obesity among adults",
        "Current cigarette smoking among adults",
        "Diabetes among adults"
    ]

    df_selected = df[df["Question"].isin(selected_questions)].copy()
    print(f"Rows with selected indicators: {len(df_selected)}")

    # Pivot: each row = state-year, columns = indicators
    wide = df_selected.pivot_table(
        index=["LocationAbbr", "YearStart"],
        columns="Question",
        values="DataValue",
        aggfunc="mean"
    ).reset_index()

    print(f"Wide table shape: {wide.shape}")

    # ----------------------------------------------------------
    # Step 6: Rename columns and drop missing
    # ----------------------------------------------------------
    col_map = {
        "LocationAbbr": "state",
        "YearStart": "year",
        "Obesity among adults": "obesity",
        "Current cigarette smoking among adults": "smoking",
        "Diabetes among adults": "diabetes"
    }
    wide = wide.rename(columns=col_map)

    # Keep required columns (including year for RQ3 COVID analysis)
    model_cols = ["state", "year", "obesity", "smoking", "diabetes"]
    df_model = wide[model_cols].dropna()

    # Add COVID flag for RQ3 (2020+ = COVID period)
    df_model["covid_flag"] = (df_model["year"] >= 2020).astype(int)

    print(f"Final modeling data: {len(df_model)} observations")
    print(f"Columns: {list(df_model.columns)}")

    # ----------------------------------------------------------
    # Step 7: Summary statistics
    # ----------------------------------------------------------
    print("\nSummary statistics:")
    print(df_model[["obesity", "smoking", "diabetes"]].describe().round(2))

    print(f"\nYears in data: {sorted(df_model['year'].unique())}")
    print(f"Pre-COVID observations: {(df_model['covid_flag'] == 0).sum()}")
    print(f"COVID observations: {(df_model['covid_flag'] == 1).sum()}")

    # ----------------------------------------------------------
    # Step 8: Save to CSV
    # ----------------------------------------------------------
    df_model.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    # Also show correlations as sanity check
    print("\nCorrelation matrix (sanity check):")
    print(df_model[["obesity", "smoking", "diabetes"]].corr().round(3))

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
