"""
Step 6: MPI4py Parallel Bootstrap for RQ1 & RQ3 Analysis

RQ1: How strongly do obesity, smoking, and diabetes co-vary?
     - Bootstrap CIs for correlations and regression coefficients

RQ3: Did COVID years create a structural break?
     - Bootstrap CIs for mean differences (pre-COVID vs COVID)

Usage:
  mpiexec -n 4 python mpi_bootstrap.py
"""

from mpi4py import MPI
import numpy as np
import pandas as pd
import time

# ============================================================
# MPI Setup
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================
# Configuration
# ============================================================
N_BOOTSTRAP = 1000          # Total number of bootstrap samples
CONFIDENCE_LEVEL = 0.95     # For confidence intervals
RANDOM_SEED_BASE = 42       # Base seed (each rank gets seed + rank)
DATA_FILE = "cdi_model_data.csv"  # Prepared modeling data

# ============================================================
# Helper Functions - RQ1 (Co-variation)
# ============================================================

def compute_correlations(data):
    """Compute pairwise correlations between obesity, smoking, diabetes."""
    corr_obesity_smoking = np.corrcoef(data['obesity'], data['smoking'])[0, 1]
    corr_obesity_diabetes = np.corrcoef(data['obesity'], data['diabetes'])[0, 1]
    corr_smoking_diabetes = np.corrcoef(data['smoking'], data['diabetes'])[0, 1]
    return corr_obesity_smoking, corr_obesity_diabetes, corr_smoking_diabetes


def compute_regression_coefs(data):
    """
    Fit OLS regression: obesity ~ smoking + diabetes
    Returns: (intercept, beta_smoking, beta_diabetes, R2)
    """
    X = np.column_stack([
        np.ones(len(data)),
        data['smoking'].values,
        data['diabetes'].values
    ])
    y = data['obesity'].values

    # OLS: beta = (X'X)^{-1} X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)

    # R-squared
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return beta[0], beta[1], beta[2], r2


# ============================================================
# Helper Functions - RQ3 (COVID Structural Break)
# ============================================================

def compute_covid_mean_diff(data):
    """
    Compute mean differences: COVID period - Pre-COVID period
    Returns dict with mean_diff for obesity, smoking, diabetes
    """
    pre_covid = data[data['covid_flag'] == 0]
    covid = data[data['covid_flag'] == 1]

    return {
        'diff_obesity': covid['obesity'].mean() - pre_covid['obesity'].mean(),
        'diff_smoking': covid['smoking'].mean() - pre_covid['smoking'].mean(),
        'diff_diabetes': covid['diabetes'].mean() - pre_covid['diabetes'].mean()
    }


def compute_covid_regression(data, indicator):
    """
    Fit OLS: indicator ~ covid_flag
    Returns: (intercept, beta_covid)
    """
    X = np.column_stack([
        np.ones(len(data)),
        data['covid_flag'].values
    ])
    y = data[indicator].values

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)

    return beta[0], beta[1]  # intercept, beta_covid


# ============================================================
# Bootstrap Functions
# ============================================================

def bootstrap_sample(data, rng):
    """Generate one bootstrap sample (resample with replacement)."""
    n = len(data)
    indices = rng.choice(n, size=n, replace=True)
    return data.iloc[indices].reset_index(drop=True)


def run_bootstrap_iteration(data, rng):
    """
    Run one bootstrap iteration for both RQ1 and RQ3.
    Returns dict with all statistics.
    """
    boot_data = bootstrap_sample(data, rng)

    # RQ1: Correlations
    corr_os, corr_od, corr_sd = compute_correlations(boot_data)

    # RQ1: Regression (obesity ~ smoking + diabetes)
    intercept, beta_smoking, beta_diabetes, r2 = compute_regression_coefs(boot_data)

    # RQ3: Mean differences (COVID - pre-COVID)
    covid_diffs = compute_covid_mean_diff(boot_data)

    # RQ3: COVID regression coefficients
    _, beta_covid_obesity = compute_covid_regression(boot_data, 'obesity')
    _, beta_covid_smoking = compute_covid_regression(boot_data, 'smoking')
    _, beta_covid_diabetes = compute_covid_regression(boot_data, 'diabetes')

    return {
        # RQ1: Correlations
        'corr_obesity_smoking': corr_os,
        'corr_obesity_diabetes': corr_od,
        'corr_smoking_diabetes': corr_sd,
        # RQ1: Regression
        'intercept': intercept,
        'beta_smoking': beta_smoking,
        'beta_diabetes': beta_diabetes,
        'r2': r2,
        # RQ3: Mean differences
        'diff_obesity': covid_diffs['diff_obesity'],
        'diff_smoking': covid_diffs['diff_smoking'],
        'diff_diabetes': covid_diffs['diff_diabetes'],
        # RQ3: COVID regression betas
        'beta_covid_obesity': beta_covid_obesity,
        'beta_covid_smoking': beta_covid_smoking,
        'beta_covid_diabetes': beta_covid_diabetes
    }


def compute_ci(values, confidence=0.95):
    """Compute percentile confidence interval."""
    alpha = 1 - confidence
    lower = np.percentile(values, 100 * alpha / 2)
    upper = np.percentile(values, 100 * (1 - alpha / 2))
    return lower, upper


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    # ----------------------------------------------------------
    # Step 1: Rank 0 loads data and broadcasts to all ranks
    # ----------------------------------------------------------
    if rank == 0:
        print("=" * 60)
        print("MPI4py Parallel Bootstrap Analysis (RQ1 + RQ3)")
        print("=" * 60)
        print(f"Number of processes: {size}")
        print(f"Total bootstrap samples: {N_BOOTSTRAP}")
        print(f"Samples per process: ~{N_BOOTSTRAP // size}")
        print()

        # Load data
        try:
            df = pd.read_csv(DATA_FILE)
            print(f"Loaded data: {len(df)} observations")
            print(f"Columns: {list(df.columns)}")
            print(f"Pre-COVID: {(df['covid_flag'] == 0).sum()}, COVID: {(df['covid_flag'] == 1).sum()}")
        except FileNotFoundError:
            print(f"ERROR: {DATA_FILE} not found!")
            print("Please run: python prepare_data.py first")
            comm.Abort(1)

        # Convert to dict for broadcasting
        data_dict = {
            'obesity': df['obesity'].values,
            'smoking': df['smoking'].values,
            'diabetes': df['diabetes'].values,
            'covid_flag': df['covid_flag'].values
        }

        start_time = time.time()
    else:
        data_dict = None
        start_time = None

    # Broadcast data to all ranks
    data_dict = comm.bcast(data_dict, root=0)
    start_time = comm.bcast(start_time, root=0)

    # Reconstruct DataFrame on each rank
    df_local = pd.DataFrame(data_dict)

    # ----------------------------------------------------------
    # Step 2: Each rank computes its share of bootstrap samples
    # ----------------------------------------------------------

    # Determine how many samples this rank computes
    samples_per_rank = N_BOOTSTRAP // size
    remainder = N_BOOTSTRAP % size

    if rank < remainder:
        my_n_samples = samples_per_rank + 1
    else:
        my_n_samples = samples_per_rank

    if rank == 0:
        print(f"\nStarting parallel bootstrap computation...")

    # Each rank uses different random seed
    rng = np.random.default_rng(RANDOM_SEED_BASE + rank)

    # Run bootstrap iterations
    local_results = []
    for i in range(my_n_samples):
        result = run_bootstrap_iteration(df_local, rng)
        local_results.append(result)

    # ----------------------------------------------------------
    # Step 3: Gather all results to rank 0
    # ----------------------------------------------------------
    all_results = comm.gather(local_results, root=0)

    # ----------------------------------------------------------
    # Step 4: Rank 0 combines results and computes CIs
    # ----------------------------------------------------------
    if rank == 0:
        elapsed = time.time() - start_time

        # Flatten list of lists
        combined_results = []
        for rank_results in all_results:
            combined_results.extend(rank_results)

        print(f"Completed {len(combined_results)} bootstrap samples in {elapsed:.2f} seconds")
        print()

        # Convert to DataFrame
        boot_df = pd.DataFrame(combined_results)

        # ----------------------------------------------------------
        # Compute original estimates
        # ----------------------------------------------------------
        orig_corrs = compute_correlations(df_local)
        orig_coefs = compute_regression_coefs(df_local)
        orig_diffs = compute_covid_mean_diff(df_local)
        _, orig_beta_covid_obesity = compute_covid_regression(df_local, 'obesity')
        _, orig_beta_covid_smoking = compute_covid_regression(df_local, 'smoking')
        _, orig_beta_covid_diabetes = compute_covid_regression(df_local, 'diabetes')

        # ----------------------------------------------------------
        # Print Results - RQ1
        # ----------------------------------------------------------
        print("=" * 60)
        print("RQ1: Co-variation of Obesity, Smoking, Diabetes")
        print("=" * 60)

        print("\n--- Pairwise Correlations ---")
        corr_names = [
            ('corr_obesity_smoking', 'Obesity vs Smoking', orig_corrs[0]),
            ('corr_obesity_diabetes', 'Obesity vs Diabetes', orig_corrs[1]),
            ('corr_smoking_diabetes', 'Smoking vs Diabetes', orig_corrs[2])
        ]

        for col, name, orig_val in corr_names:
            ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
            print(f"{name}:")
            print(f"  Point estimate: {orig_val:.4f}")
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print()

        print("--- Regression: obesity ~ smoking + diabetes ---")
        reg_names = [
            ('intercept', 'Intercept', orig_coefs[0]),
            ('beta_smoking', 'Beta (smoking)', orig_coefs[1]),
            ('beta_diabetes', 'Beta (diabetes)', orig_coefs[2]),
            ('r2', 'R-squared', orig_coefs[3])
        ]

        for col, name, orig_val in reg_names:
            ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
            print(f"{name}:")
            print(f"  Point estimate: {orig_val:.4f}")
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
            print()

        # ----------------------------------------------------------
        # Print Results - RQ3
        # ----------------------------------------------------------
        print("=" * 60)
        print("RQ3: COVID Structural Break Analysis")
        print("=" * 60)

        print("\n--- Mean Difference (COVID - Pre-COVID) ---")
        diff_names = [
            ('diff_obesity', 'Obesity', orig_diffs['diff_obesity']),
            ('diff_smoking', 'Smoking', orig_diffs['diff_smoking']),
            ('diff_diabetes', 'Diabetes', orig_diffs['diff_diabetes'])
        ]

        for col, name, orig_val in diff_names:
            ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
            sig = "***" if (ci_low > 0 or ci_high < 0) else "(not significant)"
            print(f"{name}:")
            print(f"  Mean difference: {orig_val:.4f}")
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}] {sig}")
            print()

        print("--- COVID Regression Coefficients (indicator ~ covid_flag) ---")
        covid_reg_names = [
            ('beta_covid_obesity', 'Beta COVID (obesity)', orig_beta_covid_obesity),
            ('beta_covid_smoking', 'Beta COVID (smoking)', orig_beta_covid_smoking),
            ('beta_covid_diabetes', 'Beta COVID (diabetes)', orig_beta_covid_diabetes)
        ]

        for col, name, orig_val in covid_reg_names:
            ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
            sig = "***" if (ci_low > 0 or ci_high < 0) else "(not significant)"
            print(f"{name}:")
            print(f"  Point estimate: {orig_val:.4f}")
            print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}] {sig}")
            print()

        # ----------------------------------------------------------
        # Summary
        # ----------------------------------------------------------
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total bootstrap samples: {len(combined_results)}")
        print(f"MPI processes used: {size}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Throughput: {len(combined_results) / elapsed:.1f} samples/second")
        print()

        # Save results
        output_file = "bootstrap_results.csv"
        boot_df.to_csv(output_file, index=False)
        print(f"Bootstrap samples saved to: {output_file}")

        # Save summary
        summary_file = "bootstrap_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Bootstrap Analysis Summary (RQ1 + RQ3)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"N bootstrap samples: {len(combined_results)}\n")
            f.write(f"N processes: {size}\n")
            f.write(f"Time: {elapsed:.2f} seconds\n\n")

            f.write("RQ1: Correlations (95% CI)\n")
            f.write("-" * 30 + "\n")
            for col, name, orig_val in corr_names:
                ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
                f.write(f"  {name}: {orig_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]\n")

            f.write("\nRQ1: Regression coefficients (95% CI)\n")
            f.write("-" * 30 + "\n")
            for col, name, orig_val in reg_names:
                ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
                f.write(f"  {name}: {orig_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]\n")

            f.write("\nRQ3: Mean differences COVID - Pre-COVID (95% CI)\n")
            f.write("-" * 30 + "\n")
            for col, name, orig_val in diff_names:
                ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
                sig = "***" if (ci_low > 0 or ci_high < 0) else ""
                f.write(f"  {name}: {orig_val:.4f} [{ci_low:.4f}, {ci_high:.4f}] {sig}\n")

            f.write("\nRQ3: COVID regression betas (95% CI)\n")
            f.write("-" * 30 + "\n")
            for col, name, orig_val in covid_reg_names:
                ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
                sig = "***" if (ci_low > 0 or ci_high < 0) else ""
                f.write(f"  {name}: {orig_val:.4f} [{ci_low:.4f}, {ci_high:.4f}] {sig}\n")

        print(f"Summary saved to: {summary_file}")
