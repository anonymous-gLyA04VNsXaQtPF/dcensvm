# dcensvm: Distributed Consensus Support Vector Machine

## Overview

This R package `dcensvm`  implements Distributed Consensus Support Vector Machine algorithms for high-dimensional 
classification over decentralized networks. The package provides methods for training SVMs in a distributed 
computing environment using consensus-based approaches. You need a  **SLURM cluster with R version 4.4.0** to replicate 
the results in the paper. The runing time of the simulations is about 8 hours with 100 nodes and 2 CPU cors per node running simultaneously. 
The time may vary depending on your cluster configuration and the number of jobs you can run simultaneously.

## Install Required Packages

Assume you are in a SLURM cluster with R version 4.4.0 by running the following command in the bash console:

```bash
# Load necessary modules
module load R/4.4.0
```

Assume you are in the current R project named `dcensvm` by double clicking the file `dcensvm.Rproj`. If you are in 
the correct directory, you can run the following command in the R console to check if you are in the correct directory:

```r
list.dirs(".")
#  [1] "."  "./man" "./Output"  "./Output/figs" "./Output/LOG"  "./R" "./real_data" "./Sim" "./src" "./utils" 
```


You can install the required packages by running the following command in the R console:

```r
rm(list = ls())
# Install roxygen2 and remotes  if you don't have them
if (!require("remotes")) install.packages("remotes")
if (!require("roxygen2")) install.packages("roxygen2")

source("reinstall_loaded_packages.R")
```

## Install Our R Package `dcensvm`

You can install the  package from  the current  directory by running the following command in the R console:

``` r
# Generate documentation
roxygen2::roxygenise()

# Install from local source
remotes::install_local(upgrade = "never")
```

If you have correctly configured your R environment and installed the required packages with the above commands, the 
following command running in the R console will generate outputs as follows:

``` r
library(doRNG)
library(foreach)
library(doFuture)
library(parallel)
library(tictoc)
library(MASS)
library(pracma)
library(igraph) # for graph
library(glmnet)
library(dplyr)
library(dcensvm)
library(hdsvm) # initial
library(rslurm) # submit slurm jobs in HPC
library(peakRAM) # to check memory
source("utils/sim_utils.R")
out <- simulation(n = 100, p = 20, quiet = TRUE); 
out[1:5]
# >  RMSE_Local    RMSE_Our     RMSE_DC RMSE_Pooled RMSE_deSubG 
# >  0.7765992   0.4628361   0.3657407   0.3462107   0.3350623 
```

## Replicating the Results

You need a SLURM cluster to run the simulations. We assume you have a SLURM cluster with R installed. 
The SLURM cluster should allow submitting at least 100 jobs with 2 CPUs each and 1GB of memory per job. With 100 jobs running simutaneously, the simulations will take about 8 hour to complete.
Note that the simulations will take a longer or shorter time depending on your cluster configuration and the number of jobs you can run simultaneously. 
The current CPU models are ranging from Intel Xeon E5-2650 v3 @ 2.30GHz to Intel Xeon Gold 6230 @ 2.10 GHz. 
Assume you are in the root directory of the R project `dcensvm`, you can run the following command in the shell to submit the simulations and replicate all the results mentioned in the paper:


```bash
sbatch run_all_slurm.sh
sbatch run_real_data_slurm.sh
```

NOTE: sometimes, you need add the following comments to the header of the `run_all_slurm.sh` and 
`run_real_data_slurm.sh` files to make it 
work in your SLURM cluster. 
Replace `YOUR_ACCOUNT_NAME` with your SLURM account name and `YOUR_EMAIL_ADDRESS_FOR_NOTIFICATION` with your email address to receive notifications about the job status.

```bash
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS_FOR_NOTIFICATION
#SBATCH --mail-type=BEGIN,END,FAIL
```

We use R package `rslurm` to submit the jobs in SLURM. This means you need to configure the `rslurm` package to work 
with your SLURM cluster according to the documentation [here](https://cran.r-project.org/web/packages/rslurm/vignettes/rslurm.html). In particular, you need to set the `account` to your SLURM 
account name in the template file `/home/USERNAME/R/x86_64-pc-linux-gnu-library/4.4/rslurm/templates/submit_sh.txt`
as follows:

```bash
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS_FOR_NOTIFICATION
#SBATCH --mail-type=BEGIN,END,FAIL
```


In addition, you need specify a correct R version by modifying the following line based on you SLURM cluster 
configuration:
```bash
# Load necessary modules
module load R/4.4.0
```

Note you can search the available R versions by running the following command in the terminal:
```bash
module avail R
```

## Features

  - Implementation of distributed consensus SVM algorithms
  - Support for high-dimensional data classification
  - Various kernel functions
  - Simulation utilities for testing and benchmarking
  - Parameter tuning using BIC and other methods

## Usage

``` r
library(dcensvm)

# Basic usage example
out <- simulation(n = 100, p = 20, quiet = TRUE); 
out[1:5]
# >  RMSE_Local    RMSE_Our     RMSE_DC RMSE_Pooled RMSE_deSubG 
# >  0.7765992   0.4628361   0.3657407   0.3462107   0.3350623 
```
---


## Project Structure

  - `R/`: R function files
    
      - `consensus_DC.R`: Aggregates the local SVM estimates by averaging them across nodes, employing the consensus protocol (method `Avg.`)
      - `bic_hdsvm.R`: Parameter estimate using BIC for initial and pooled estimates. 
      - `utils.R`: Utility functions for data generation, F1 score, etc.
      - `fixRNGStream.R`: RNG stream management for reproducibility.
      - `dcensvm-package.R`: Package-level documentation and registration.
      - `RcppExports.R`: Auto-generated exports for C++ functions.


  
  - `src/`: C++ source code integrated with R using Rcpp

      - `decensvm.cpp`: Implements our **deCSVM** algorithm with BIC-based $\lambda$ selection, and one can use **'dcensvm::decentralizedsvm_cpp()'** in R to use it.
      - `decensvm_lambda.cpp`: Fixed $\lambda$ version of our deCSVM algorithm algorithm for known regularization parameters. Key functions: `decentralizedsvm_lambda()`in R.
      - `decensvm_real.cpp`: Real-data optimized version supporting different sample sizes. Key functions: `decentralizedsvm_cpp_real()`in R.
      - `deSubGD.cpp`: Subgradient descent-based method. Key functions: `deSubGD_svm()`in R.
      - `deSubGD_real.cpp`: Real-data optimized subgradient descent method. Key functions: `decentralizedsvm_cpp_real()`in R.
      - `utils.cpp`: Support functions, core utilities and mathematical operations.
          - Key functions:
          - `pmax_arma()`: R-compatible pmax implementation;
          - `soft_thresholding_cpp()`: L1 proximal operator;
          - `calN_j_cpp()`: Index calculation for distributed samples.
      - `decentralizedsvm.h`: Header file, shared declarations and helper functions.



  - `utils/`: Additional utilities for testing and simulation.
    
      - `sim_utils.R`: Helper functions for simulation.


  - `Sim/`: Simulation code for testing performance. 

      - `sim_plot_types_of_loss_slurm.R`:  Plot codes of Figure A.1 in Supplementary Material.

  
  - `Output/`: Directory for storing **simulation results** from the various simulations.


---
## Simulations

Below is a detailed explanation of each simulation script and its purpose. All scripts follow the same execution 
pattern: parameter setup $\rightarrow$ SLURM job submission $\rightarrow$ result aggregation $\rightarrow$ output 
generation. In all simulation setting except Section 4.2, we vary the correlation parameter $\rho$ in the covariance 
matrix of covariate vector $x$ over the values $\{0.3, 0.5, 0.7, 0.9\}$. Expect 4.2, 4.3, 4.4, and K.4, we fix the 
number of nodes to $m = 10$, the local sample size to $n = 200$ in the other cases. 

  - The proposed method: 
      - `deCSVM`, decentralized penalized convoluted support vector machine: `dcensvm::decentralizedsvm_cpp()`;
  - Compared with four methods:
      - `Pooled`, which computes the $\ell_1$-penalized SVM estimate using the entire dataset and serves as the benchmark: `bic.hdsvm()`;
      - `Local`, where each node independently computes its own $\ell_1$-penalized SVM estimate using only the local data available at that node: `bic.hdsvm()`;
      - `D-subGD`, where nodes collaboratively solve the objective function with subgradient descent: `deSubGD_svm()`;
      - `Avg`, which aggregates the local SVM estimates by averaging them across nodes, employing the consensus protocol: `consensus_DC()`;




### 4.2 Effect of Iterations
  - `sim_iteration_slurm.R`: Different kernels converge with the number of iterations when $(n,p) = (100,50)$ and $(n,p) = (200,100)$. Results store in `Output/figs/fig_iterations_kernel_type_p_50.pdf` and `Output/figs/fig_iterations_kernel_type_p_100.pdf`. 

### 4.3 Effect of Local Sample Size
  - `sim_local_sample_size_slurm.R`: Effect of local sample size for varying local sample sizes and the dimensions $(n, p)$ in $\{(100, 100), (200, 100), (200, 200)\}$. Results store in `Output/np_result.tex`. 


### 4.4 Effect of Topology  
  - `sim_m_slurm.R`: Number of machines/nodes analysis that fix the total sample size to $N = 4000$ and vary the 
    number of 
    nodes $m \in \{5, 10, 20\}$. Results store in `Output/m_result.tex`. 
  - `sim_pc_slurm.R`: Effect of network sparsity by varying the network connection probability $p_c \in \{0.3, 0.5, 
    0.8\}$. For this analysis, 
    we fix the number of nodes to $m = 10$, the local sample size to $n = 200$, and the dimensionality to $p = 
    100$. Results store in `Output/pc_result.tex`. 

### 4.5 Effect of Sign Flips 
  - `sim_flips_slurm.R`: Analysis of label flips. We set the label flip proportion $p_{\rm flip} \in \{0.01, 0.05, 0.1\}
    $. Results store in `Output/flips_result.tex`.  
   

### K.1 Effect of Kernel Type
  - `sim_kernel_slurm.R`: Robustness of our deCSVM across various kernel smoothing techniques, including uniform, Laplacian, logistic, Gaussian, and Epanechnikov kernels with kernel_type = 1-5 respectively. Results store in `Output/kernel_result.tex`.

### K.2 Effect of the Tuning Parameter $\lambda$
  - `sim_bic_plot_slurm.R`: Compares error and estimated sparsity level under different kernel types (kernel_type 1-5) using BIC criterion with different $\lambda$ in Figure K.3. Results store in `Output/plot_lambda_vs_error.pdf` and `Output/plot_lambda_vs_rank.pdf`.

### K.3 Sensitivity Analysis for the Bandwidth $h$
  - `sim_bandwidth_slurm.R`: Bandwidth parameter tuning analysis that $h = \max\{0.05, C (\log p / N)^{1 / 4}\}$ with $C$ in $\{0.1, 0.5, 1.0, 5.0\}$. Results store in `Output/bandwidth_result.tex`. 

### K.4 Effect of the Tuning Parameter $\rho_\ell$
  - `sim_C_rho_slurm.R`: Sensitivity analysis for $\rho_\ell = (1 + \delta)c_h \Lambda_{\max}( \frac{1}{n} \sum_{i \in \mathcal{I}_\ell} x_i x_i^\top)$ with $\delta \in \{1.01, 2, 5\}$ in Figure K.2. Results store in `Output/dimension_increased_result.tex`. 

### K.5 Effect of Augmented Parameter $\tau$
  - `sim_tau_slurm.R`:Evaluate the performance of deCSVM with Lagrangian penalty parameter $\tau \in \{0.5, 1, 2, 4\}
    $. Results store in `Output/figs/fig_iterationstau_0.3.pdf` and `Output/figs/fig_iterationstau_0.5.pdf`.

  
### K.6 Effect of Dimension  
  - `sim_dimension_increase_slurm.R`:  Assess the performance of our deCSVM method under increasing variable dimensionality with $p \in (100, 200, 300, 400, 500)$. Results store in `Output/dimension_increased_result.tex`. 

### K.7 Effect of the Sparsity $s$
  - `sim_sparsity_slurm.R`: Evaluate the performance of deCSVM and baseline methods under varying sparsity levels $s \in \{5, 10, 15, 20, 30\}$. Results store in `Output/sparsity_result.tex`. 

### K.8 Effect of the Sample Size $N$   
  - `sim_sample_size_slurm.R`: Sample size effect analysis that we fix the number of nodes to $m = 10$ and the variable dimension to $p = 100$, and consider a fully connected communication network with connection probability $p_c = 1$. The local sample size $n \in \{100, 200, 400\}$ varies per node. Results store in `Output/sample_size_result.tex`.

### Figure 1
- `sim_plot_types_of_loss_slurm.R`: Plot Figure 1 in the paper. Results store in `Output/figs/hinge_loss_plot.
pdf` and `Output/figs/hinge_loss_plot.pdf`.

---

## Real-Data Example

  - `crime_data_clean.R`: This script processes and cleans the Communities and Crime dataset (https://archive.ics.uci.edu/dataset/183/communities+and+crime) for analysis. 
    - Specifically, we read raw Excel data 
      (`communities_and_crime.xlsx`), apply attribute names from `attributes.csv`, handle missing values (NA, NaN, 
      NULL, "?"), and drops columns with missing values except core features. 
    - We convert state codes to 9 US Census divisions (Pacific, Mountain, West North Central, West South Central, East North Central, East South Central, South Atlantic, Middle Atlantic, New England) and create numeric division labels (1-9).  
    - We calculate median violent crime rate and creates binary label `risk_level` (1 = high risk, -1 = low risk). 
    - Finally, we save cleaned dataset to `crime_cleaned.csv`. 
 
-  `real_data_slurm.R`: This script performs 5 methods on the cleaned crime data with 100 replications.
    - Loads cleaned data (`crime_cleaned.csv`);
    - Splits data into training (80%) and testing (20%) sets;
    - Introduces label flipped noise (`flip_prop` $\in$ {0, 0.01, 0.02});
    - Evaluates accuracy on test set and mean support size (non-zero coefficients).
- Results store in `Output/sim_real_data.tex`. 


---
<!--
## Dependencies

  - Rcpp
  - RcppArmadillo
  - doFuture
  - parallel
  - MASS
  - igraph
  - glmnet
  - Other packages listed in DESCRIPTION

## Version Requirements

  - **Package Version**: 0.1.0
  - **R Version**: R = 4.4.0
  - **Key Dependencies**:
      - Rcpp (\>= 1.0.0)
      - RcppArmadillo (\>= 0.10.0)
      - doFuture (\>= 1.1.0)
      - igraph (\>= 1.2.6)
      - glmnet (\>= 4.1.0)
      - MASS (\>= 7.3.0)
-->

## Session Information

The following will provide details about the R version, loaded packages, and
platform information that may affect simulation results.

``` r
sessioninfo::session_info(info = "all", to_file = "session_info.txt")
```

The `session_info.txt` reads as follows:

```text
─ Session info ───────────────────────────────────────────────────────────────────────────────────────
 setting  value
 version  R version 4.4.0 (2024-04-24)
 os       Rocky Linux 8.7 (Green Obsidian)
 system   x86_64, linux-gnu
 ui       RStudio
 language (EN)
 collate  en_US.UTF-8
 ctype    en_US.UTF-8
 tz       America/Detroit
 date     2025-05-31
 rstudio  1.3.1056  (server)
 pandoc   2.0.6 @ /usr/bin/pandoc

─ Packages ───────────────────────────────────────────────────────────────────────────────────────────
 ! package       * version    date (UTC) lib source
   abind           1.4-5      2016-07-21 [2] CRAN (R 4.4.0)
   backports       1.5.0      2024-05-23 [2] CRAN (R 4.4.0)
   brio            1.1.5      2024-04-24 [2] CRAN (R 4.4.0)
   broom           1.0.6      2024-05-17 [2] CRAN (R 4.4.0)
   callr           3.7.6      2024-03-25 [2] CRAN (R 4.4.0)
   car             3.1-2      2023-03-30 [2] CRAN (R 4.4.0)
   carData         3.0-5      2022-01-06 [2] CRAN (R 4.4.0)
   cellranger      1.1.0      2016-07-27 [2] CRAN (R 4.4.0)
   cli             3.6.3      2024-06-21 [2] CRAN (R 4.4.0)
   codetools       0.2-20     2024-03-31 [2] CRAN (R 4.4.0)
   colorspace      2.1-0      2023-01-23 [2] CRAN (R 4.4.0)
   crayon          1.5.3      2024-06-20 [2] CRAN (R 4.4.0)
   data.table      1.16.0     2024-08-27 [2] CRAN (R 4.4.0)
 P dcensvm       * 0.1.0      2025-05-27 [?] local (/home/USERNAME/r_home/dcensvm)
   desc            1.4.3      2023-12-10 [2] CRAN (R 4.4.0)
   digest          0.6.37     2024-08-19 [1] CRAN (R 4.4.0)
   doFuture      * 1.1.0      2025-05-22 [1] CRAN (R 4.4.0)
   doParallel    * 1.0.17     2022-02-07 [2] CRAN (R 4.4.0)
   doRNG         * 1.8.6.2    2025-04-02 [1] CRAN (R 4.4.0)
   dplyr         * 1.1.4      2023-11-17 [2] CRAN (R 4.4.0)
   fansi           1.0.6      2023-12-08 [2] CRAN (R 4.4.0)
   foreach       * 1.5.2      2022-02-02 [2] CRAN (R 4.4.0)
   future        * 1.49.0     2025-05-09 [1] CRAN (R 4.4.0)
   future.apply  * 1.11.3     2024-10-27 [2] CRAN (R 4.4.0)
   generics        0.1.3      2022-07-05 [2] CRAN (R 4.4.0)
   ggplot2       * 3.5.1      2024-04-23 [2] CRAN (R 4.4.0)
   ggpubr        * 0.6.0      2023-02-10 [2] CRAN (R 4.4.0)
   ggsignif        0.6.4      2022-10-13 [2] CRAN (R 4.4.0)
   glmnet        * 4.1-8      2023-08-22 [2] CRAN (R 4.4.0)
   globals         0.18.0     2025-05-08 [1] CRAN (R 4.4.0)
   glue            1.7.0      2024-01-09 [2] CRAN (R 4.4.0)
   gtable          0.3.5      2024-04-22 [2] CRAN (R 4.4.0)
   hdsvm         * 1.0.1      2025-02-11 [1] CRAN (R 4.4.0)
   igraph        * 2.1.4      2025-01-23 [1] CRAN (R 4.4.0)
   iterators     * 1.0.14     2022-02-05 [2] CRAN (R 4.4.0)
   knitr           1.47       2024-05-29 [2] CRAN (R 4.4.0)
   lattice         0.22-6     2024-03-20 [2] CRAN (R 4.4.0)
   lifecycle       1.0.4      2023-11-07 [2] CRAN (R 4.4.0)
   listenv         0.9.1      2024-01-29 [2] CRAN (R 4.4.0)
   magrittr        2.0.3      2022-03-30 [2] CRAN (R 4.4.0)
   MASS          * 7.3-60.2   2024-04-26 [1] CRAN (R 4.4.0)
   Matrix        * 1.7-0      2024-04-26 [1] CRAN (R 4.4.0)
   munsell         0.5.1      2024-04-01 [2] CRAN (R 4.4.0)
   parallelly      1.44.0     2025-05-07 [1] CRAN (R 4.4.0)
   peakRAM       * 1.0.2      2017-01-16 [1] CRAN (R 4.4.0)
   pillar          1.9.0      2023-03-22 [2] CRAN (R 4.4.0)
   pkgbuild        1.4.4      2024-03-17 [2] CRAN (R 4.4.0)
   pkgconfig       2.0.3      2019-09-22 [2] CRAN (R 4.4.0)
   pkgload         1.4.0      2024-06-28 [2] CRAN (R 4.4.0)
   plyr            1.8.9      2023-10-02 [2] CRAN (R 4.4.0)
   pracma        * 2.4.4      2023-11-10 [2] CRAN (R 4.4.0)
   processx        3.8.4      2024-03-16 [2] CRAN (R 4.4.0)
   ps              1.7.7      2024-07-02 [2] CRAN (R 4.4.0)
   purrr         * 1.0.2      2023-08-10 [2] CRAN (R 4.4.0)
   R6              2.5.1      2021-08-19 [2] CRAN (R 4.4.0)
   Rcpp          * 1.0.12     2024-01-09 [1] CRAN (R 4.4.0)
   RcppArmadillo * 0.12.8.3.0 2024-05-08 [1] CRAN (R 4.4.0)
   readxl        * 1.4.3      2023-07-06 [2] CRAN (R 4.4.0)
   remotes       * 2.5.0      2024-03-17 [2] CRAN (R 4.4.0)
   reshape2      * 1.4.4      2020-04-09 [2] CRAN (R 4.4.0)
   rlang           1.1.4      2024-06-04 [2] CRAN (R 4.4.0)
   rngtools      * 1.5.2      2021-09-20 [2] CRAN (R 4.4.0)
   roxygen2      * 7.3.1      2024-01-22 [1] CRAN (R 4.4.0)
   rprojroot       2.0.4      2023-11-05 [2] CRAN (R 4.4.0)
   rslurm        * 0.6.2      2023-02-24 [1] CRAN (R 4.4.0)
   rstatix         0.7.2      2023-02-01 [2] CRAN (R 4.4.0)
   scales          1.3.0      2023-11-28 [2] CRAN (R 4.4.0)
   sessioninfo     1.2.2      2021-12-06 [2] CRAN (R 4.4.0)
   shape           1.4.6.1    2024-02-23 [2] CRAN (R 4.4.0)
   stringi         1.8.4      2024-05-06 [2] CRAN (R 4.4.0)
   stringr       * 1.5.1      2023-11-14 [2] CRAN (R 4.4.0)
   survival        3.7-0      2024-06-05 [2] CRAN (R 4.4.0)
   testthat        3.2.1.1    2024-04-14 [2] CRAN (R 4.4.0)
   tibble          3.2.1      2023-03-20 [2] CRAN (R 4.4.0)
   tictoc        * 1.2.1      2024-03-18 [2] CRAN (R 4.4.0)
   tidyr         * 1.3.1      2024-01-24 [2] CRAN (R 4.4.0)
   tidyselect    * 1.2.1      2024-03-11 [2] CRAN (R 4.4.0)
   utf8            1.2.4      2023-10-22 [2] CRAN (R 4.4.0)
   vctrs           0.6.5      2023-12-01 [2] CRAN (R 4.4.0)
   whisker         0.4.1      2022-12-05 [2] CRAN (R 4.4.0)
   withr           3.0.0      2024-01-16 [2] CRAN (R 4.4.0)
   xfun            0.45       2024-06-16 [2] CRAN (R 4.4.0)
   xml2            1.3.6      2023-12-04 [2] CRAN (R 4.4.0)
   xtable        * 1.8-4      2019-04-21 [2] CRAN (R 4.4.0)

 [1] /home/USERNAME/R/x86_64-pc-linux-gnu-library/4.4
 [2] /mnt/biostat/software/CentOS/8.7/R/4.4.0/lib64/R/library

 P ── Loaded and on-disk path mismatch.

─ External software ──────────────────────────────────────────────────────────────────────────────────
 setting        value
 cairo          1.15.12
 cairoFT
 pango          1.42.3
 png            1.6.34
 jpeg           6.2
 tiff           LIBTIFF, Version 4.0.9
 tcl            8.6.8
 curl           7.61.1
 zlib           1.2.12
 bzlib          1.0.6, 6-Sept-2010
 xz             5.2.4
 deflate
 PCRE           10.32 2018-09-10
 ICU            60.3
 TRE            TRE 0.8.0 R_fixes (BSD)
 iconv          glibc 2.28
 readline       7.0
 BLAS           /mnt/biostat/software/CentOS/8.7/R/4.4.0/lib64/R/lib/libRblas.so
 lapack         /mnt/biostat/software/CentOS/8.7/R/4.4.0/lib64/R/lib/libRlapack.so
 lapack_version 3.12.0

─ Python configuration ───────────────────────────────────────────────────────────────────────────────
 Python is not available

──────────────────────────────────────────────────────────────────────────────────────────────────────
```

