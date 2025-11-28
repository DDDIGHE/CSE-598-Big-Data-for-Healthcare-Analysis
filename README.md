# AMS-598-Big-Data-for-Healthcare-Analysis

This repository contains the code and documentation for the final project of AMS 598.

## Files

| File | Description |
|------|-------------|
| `U.S._Chronic_Disease_Indicators.csv` | Raw CDC chronic disease data |
| `ams_598_colab.ipynb` | Jupyter notebook for Steps 1-5 (data cleaning, EDA, baseline modeling) |
| `5. Baseline modeling (non-MPI)_summary.md` | Summary report for Step 5 |
| `prepare_data.py` | Data preparation script for MPI analysis |
| `mpi_bootstrap.py` | MPI4py parallel bootstrap script (Step 6) |
| `run_bootstrap.slurm` | SLURM job submission file for SeaWulf |
| `6. MPI4py parallel implementation_summary.md` | Summary report for Step 6 |
| `bootstrap_results.csv` | Bootstrap samples output (generated on HPC) |
| `bootstrap_summary.txt` | Bootstrap confidence intervals summary (generated on HPC) |

## HPC Usage (Step 6)

### Files to upload to SeaWulf:
```
U.S._Chronic_Disease_Indicators.csv
prepare_data.py
mpi_bootstrap.py
run_bootstrap.slurm
```

### Commands to run on SeaWulf:
```bash
# 1. Prepare the modeling data
python prepare_data.py

# 2. CHANGE THE WORK DIR!

# 3. Submit the MPI job
sbatch run_bootstrap.slurm
```

## Procedure

- **1. Data download & documentation (10%): finished in group meeting 1**  
    

Download the U.S. Chronic Disease Indicators data, list available years/states/indicators, and write a short dataset description.  

- **2. Data cleaning & merging (15%) - Jaya Chandra & Vasanth**  
    

Handle missing/invalid values and merge files into one clean table organized by state, year, and selected indicators.  

- **3. Exploratory analysis & indicator selection (15%) - Supraja, Naveena & Poojitha**   
    

Do basic plots and summary stats, then choose a small set of key chronic disease and risk-factor indicators for the project.  

- **4. Research questions & methods design (10%) -** **Jaya Chandra & Vasanth**  
    

Propose 1–2 concrete questions we want to answer with this dataset and match them to specific analysis / modeling methods. Mathematically define the question. 

- **5. Baseline modeling (non-MPI) (15%) - Vishnu**   
    

Implement baseline models on a single machine (e.g., regression/classification/cluster analysis) and compute evaluation metrics.  

- **6. MPI4py parallel implementation (15%):LU WEI**  
    

Take part of the computation (e.g., model training, grid search, or large-scale statistics) and re-implement it with MPI4py in a parallel way.  

- **7. Results & public-health interpretation (10%) -** **Supraja, Naveena & Poojitha**   
    

Summarize the main quantitative results and explain what they mean for chronic disease patterns across states and years.  

- **8. Visualization & figure polishing (5%) -** **Supraja, Naveena & Poojitha**   
    

Produce clear, publication-style plots (maps, trends, comparisons) and make them ready to drop into the slides.  

- **9. Slides & presentation (5%) -** **Jaya Chandra & Vasanth**  
    

Assemble the final slides, write the “project procedure & participation” page, and coordinate who presents which part.