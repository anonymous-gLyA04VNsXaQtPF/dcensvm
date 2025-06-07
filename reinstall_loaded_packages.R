install_version_if_needed <- function(pkg, version, upgrade = "never", ...) {
  # Ensure R 4.4.0 is in use
  if (R.version$major != "4" || R.version$minor != "4.0") {
    stop("This installer and our replications are intended only for R 4.4.0.")
  }

  # Default user library (R's first path)
  default_lib <- .libPaths()[1]

  # Test if it's writable
  test_file <- file.path(default_lib, ".perm_test")
  writable <- tryCatch({
    file.create(test_file)
  }, warning = function(w)
    FALSE, error = function(e)
      FALSE)
  if (file.exists(test_file))
    file.remove(test_file)

  # If not writable, fall back to custom user library (for all R 4.4.x versions)
  if (!writable) {
    lib <- file.path(Sys.getenv("HOME"),
                     "R",
                     "x86_64-pc-linux-gnu-library",
                     "4.4")
    message("Default library not writable. Using fallback: ", lib)
  } else {
    lib <- default_lib
  }

  # Ensure the directory exists
  if (!dir.exists(lib)) {
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  }

  # Ensure R uses this library
  .libPaths(c(lib, .libPaths()))

  # Check if the package is already installed in any lib
  if (pkg %in% rownames(installed.packages())) {
    current_version <- as.character(packageVersion(pkg))
    if (current_version == version) {
      message(pkg, " version ", version, " is already installed. Skipping.")
      return(invisible(TRUE))
    }
  }

  # Try installing safely
  message("Installing ", pkg, " version ", version, " to ", lib)
  remotes::install_version(pkg,
                           version = version,
                           lib = lib,
                           upgrade = upgrade,
                           ...)
}


install_version_if_needed("Rcpp", version = "1.0.12")
install_version_if_needed("RcppArmadillo", version = "0.12.8.3.0")
install_version_if_needed("peakRAM", version = "1.0.2")
install_version_if_needed("hdsvm", version = "1.0.1")
install_version_if_needed("dplyr", version = "1.1.4")
install_version_if_needed("igraph", version = "2.1.4")
install_version_if_needed("pracma", version = "2.4.4")
install_version_if_needed("future", version = "1.49.0", upgrade = "always")
install_version_if_needed("doFuture", version = "1.1.0", upgrade = "always")
install_version_if_needed("doRNG", version = "1.8.6.2")
install_version_if_needed("rngtools", version = "1.5.2")
install_version_if_needed("rslurm", version = "0.6.2")
install_version_if_needed("glmnet", version = "4.1.8")
install_version_if_needed("Matrix", version = "1.7.0")
install_version_if_needed("MASS", version = "7.3.60.2")
install_version_if_needed("tictoc", version = "1.2.1")
install_version_if_needed("foreach", version = "1.5.2")
install_version_if_needed("readxl", version = "1.4.3")
install_version_if_needed("reshape2", version = "1.4.4")
install_version_if_needed("ggplot2", version = "3.5.1")
install_version_if_needed("purrr", version = "1.0.2")
install_version_if_needed("tidyr", version = "1.3.1")
install_version_if_needed("ggpubr", version = "0.6.0")
install_version_if_needed("parallel", version = "4.4.0")
# install_version_if_needed("remotes", version = "2.5.0")
# install_version_if_needed("roxygen2", version = "7.3.1")
install_version_if_needed("stringr", version = "1.5.1")
install_version_if_needed("tidyselect", version = "1.2.1")
install_version_if_needed("xtable", version = "1.8.4")
install_version_if_needed("doParallel", version = "1.0.17")
install_version_if_needed("future.apply", version = "1.11.3")


# Generate documentation
roxygen2::roxygenise()
# Install from local source
remotes::install_local(upgrade = "never")