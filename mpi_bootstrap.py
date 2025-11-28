"""
Step 6: MPI4py Parallel Bootstrap for RQ1 Analysis
===================================================
Computes bootstrap confidence intervals for:
  - Correlation coefficients (obesity-smoking, obesity-diabetes, smoking-diabetes)
  - Regression coefficients (obesity ~ smoking + diabetes)

Usage:
  mpiexec -n 4 python mpi_bootstrap.py

Author: Lu Wei
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
# Helper Functions
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


def bootstrap_sample(data, rng):
    """Generate one bootstrap sample (resample with replacement)."""
    n = len(data)
    indices = rng.choice(n, size=n, replace=True)
    return data.iloc[indices].reset_index(drop=True)


def run_bootstrap_iteration(data, rng):
    """
    Run one bootstrap iteration:
    Returns dict with correlations and regression coefficients.
    """
    boot_data = bootstrap_sample(data, rng)

    corr_os, corr_od, corr_sd = compute_correlations(boot_data)
    intercept, beta_smoking, beta_diabetes, r2 = compute_regression_coefs(boot_data)

    return {
        'corr_obesity_smoking': corr_os,
        'corr_obesity_diabetes': corr_od,
        'corr_smoking_diabetes': corr_sd,
        'intercept': intercept,
        'beta_smoking': beta_smoking,
        'beta_diabetes': beta_diabetes,
        'r2': r2
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
        print("MPI4py Parallel Bootstrap Analysis")
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
        except FileNotFoundError:
            print(f"ERROR: {DATA_FILE} not found!")
            print("Please run: python prepare_data.py first")
            comm.Abort(1)

        # Convert to dict for broadcasting (pandas doesn't broadcast well)
        data_dict = {
            'obesity': df['obesity'].values,
            'smoking': df['smoking'].values,
            'diabetes': df['diabetes'].values
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

    # Distribute remainder among first 'remainder' ranks
    if rank < remainder:
        my_n_samples = samples_per_rank + 1
        my_start = rank * (samples_per_rank + 1)
    else:
        my_n_samples = samples_per_rank
        my_start = remainder * (samples_per_rank + 1) + (rank - remainder) * samples_per_rank

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

        # Convert to DataFrame for easy analysis
        boot_df = pd.DataFrame(combined_results)

        # ----------------------------------------------------------
        # Compute original (non-bootstrap) estimates
        # ----------------------------------------------------------
        orig_corrs = compute_correlations(df_local)
        orig_coefs = compute_regression_coefs(df_local)

        # ----------------------------------------------------------
        # Print Results
        # ----------------------------------------------------------
        print("=" * 60)
        print("RESULTS: Bootstrap Confidence Intervals")
        print("=" * 60)

        # Correlations
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

        # Regression coefficients
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
        # Summary statistics
        # ----------------------------------------------------------
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total bootstrap samples: {len(combined_results)}")
        print(f"MPI processes used: {size}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Throughput: {len(combined_results) / elapsed:.1f} samples/second")
        print()

        # Save results to CSV
        output_file = "bootstrap_results.csv"
        boot_df.to_csv(output_file, index=False)
        print(f"Bootstrap samples saved to: {output_file}")

        # Save summary to text file
        summary_file = "bootstrap_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Bootstrap Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"N bootstrap samples: {len(combined_results)}\n")
            f.write(f"N processes: {size}\n")
            f.write(f"Time: {elapsed:.2f} seconds\n\n")

            f.write("Correlations (95% CI):\n")
            for col, name, orig_val in corr_names:
                ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
                f.write(f"  {name}: {orig_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]\n")

            f.write("\nRegression coefficients (95% CI):\n")
            for col, name, orig_val in reg_names:
                ci_low, ci_high = compute_ci(boot_df[col], CONFIDENCE_LEVEL)
                f.write(f"  {name}: {orig_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]\n")

        print(f"Summary saved to: {summary_file}")
