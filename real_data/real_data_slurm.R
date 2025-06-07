# This script performs 5 methods on the cleaned crime data with 100 replications under slurm jobs in HPC.
rm(list = ls())
simulation_name <- "real_data"
# ============================================================== #
# LOAD LIBRARY
# ============================================================== #
library(readxl)
library(doRNG)
library(foreach)
library(doFuture)
library(doParallel)
library(parallel)
library(dcensvm)
library(hdsvm)
library(igraph)
library(tidyr)
library(dplyr)
library(tidyr)
library(purrr)
library(rslurm) # submit slurm jobs in HPC
library(peakRAM) #  to check memory
library(igraph)

real_one_batch <- function(batch = 1, flip_prop = 0, dat_crime, adjacency_matrix, region_mapping) {
  fixRNGStream(batch)
  # ignore warnings
  options(warn = -1)

  # === Data preparation === #
  columns_drop <- c('state', 'county', 'community', 'communityname', 'fold')
  data <- dat_crime[, !(colnames(dat_crime) %in% columns_drop)]
  data$division <- as.factor(data$division)
  data$division_label <- region_mapping[data$division]

  split_data <- data %>%
    group_by(division_label) %>%
    group_modify(~ {
      n_train <- floor(0.8 * nrow(.x))
      shuffled_data <- .x[sample(nrow(.x)), ]
      training_data <- shuffled_data[1:n_train, ]
      testing_data <- shuffled_data[(n_train + 1):nrow(.x), ]
      training_data <- training_data %>% mutate(set = "training")
      testing_data <- testing_data %>% mutate(set = "testing")
      bind_rows(training_data, testing_data)
    })

  training_data <- split_data %>% filter(set == "training") %>% dplyr::select(-set)
  testing_data  <- split_data %>% filter(set == "testing") %>% dplyr::select(-set)
  training_data$risk_level_1 <- ifelse(training_data$risk_level == 1, 1, -1)

  flipped <- TRUE  # Set to TRUE or FALSE as needed
  if (flipped) {
    training_data <- training_data %>%
      group_by(division_label) %>%
      group_modify(~ {
        n_flip <- floor(flip_prop * nrow(.x))
        if (n_flip > 0) {
          flip_indices <- sample(1:nrow(.x), n_flip)
          .x$risk_level_1[flip_indices] <- -.x$risk_level_1[flip_indices]
        }
        .x
      })
  }

  p <- 99
  m <- 9
  N <- nrow(data)

  B_init <- matrix(0, p + 1, m)
  for (j in 1:9) { #j <- 1
    idx <- which(training_data$division_label == j)
    out <- dcensvm::bic.hdsvm(cbind(1,as.matrix(training_data[idx,1:99])), training_data$risk_level_1[idx], nlambda = 100, isinit = TRUE)
    B_init[, j] <- out
  }

  B_pooled <- dcensvm::bic.hdsvm(cbind(1,as.matrix(training_data[1:99])), training_data$risk_level_1, nlambda = 100, isinit = FALSE)

  out_beta_desvm <- dcensvm::decentralizedsvm_cpp_real(
    X = cbind(1, as.matrix(training_data[, 1:99])),
    y = training_data$risk_level_1,
    divisions = training_data$division_label,
    adjacency_matrix = adjacency_matrix,
    B_init = B_init,
    betaT = B_pooled,
    type = 5,
    h = 0.2,
    C_N = sqrt(log(N)),
    T_outer = 200,
    c0 = 0.013,
    tau_penalty_factor = 2,
    lambda_factor = 1e-4,
    nlambda = 100,
    lambda_max = 0.3,
    lambda_0 = 0,
    C_rho = 1.1,
    quiet = F,
    eps = 1e-5
  )
  B_desvm <- out_beta_desvm$B


  B_DC <- consensus_DC(B_init, adjacency_matrix, 200)

  out_deSubGD <- deSubGD_svm_real(
    X = cbind(1, as.matrix(training_data[, 1:99])),
    y = training_data$risk_level_1,
    adjacency_matrix,
    matrix(rnorm((p+1)*m), (p+1), m),
    divisions = training_data$division_label,
    T = 200,
    lambda_max = 20,
    eps = 1e-8,
    quiet = TRUE
  )

  yy <- ifelse(testing_data$risk_level == 1, 1, -1)
  y_pred_desvm <- y_pred_init <- y_pred_subGD <- y_pred_DC <- numeric(nrow(testing_data))

  for (j in 1:m) {
    idx <- which(testing_data$division_label == j)
    X_test <- as.matrix(testing_data[idx, 1:99])
    y_pred_desvm[idx] <- sign(cbind(1, X_test) %*% B_desvm[, j])
    y_pred_init[idx] <- sign(cbind(1, X_test) %*% B_init[, j])
    y_pred_subGD[idx] <- sign(cbind(1, X_test) %*% out_deSubGD$B[, j])
    y_pred_DC[idx] <- sign(cbind(1, X_test) %*% B_DC[, j])
  }

  accuracy_desvm <- mean(yy == y_pred_desvm)
  accuracy_pooled <- mean(yy == sign(cbind(1, as.matrix(testing_data[, 1:99])) %*% B_pooled))
  accuracy_init <- mean(yy == y_pred_init)
  accuracy_subGD <- mean(yy == y_pred_subGD)
  accuracy_DC <- mean(yy == y_pred_DC)

  return(c(accuracy_desvm, accuracy_pooled, accuracy_init, accuracy_subGD, accuracy_DC,
           mean(colSums(B_desvm != 0)),  sum(B_pooled!=0), mean(colSums(B_init != 0)), mean(colSums(out_deSubGD$B !=
                                                                                                      0)), mean
           (colSums(B_DC != 0)),
  flip_prop = flip_prop))
}




# dat_crime <- read_excel("real_data/crime_cleaned.xlsx", na = c("", "NA", "NaN", "NULL", "?"))
dat_crime <- read.csv("real_data/crime_cleaned.csv", na.strings = c("", "NA", "NaN", "NULL", "?"))
columns_drop <- c('state', 'county', 'community', 'communityname', 'fold')
data <- dat_crime[,!(colnames( dat_crime)%in% columns_drop)]

regions <- c("Pacific", "Mountain", "West North Central",
             "West South Central", "East North Central",
             "East South Central", "South Atlantic",
             "Middle Atlantic", "New England")

region_mapping <- setNames(1:9, regions)
data$division <- as.factor(data$division)
levels(data$division)

# Assign corresponding label numbers to regions in the data frame
data$division_label <- region_mapping[data$division]

# Define connectivity
# Redefine edges according to the connection graph
edges <- matrix(c(
  1, 2,  # Pacific - Mountain
  2, 3,  # Mountain - West North Central
  2, 4,  # Mountain - West South Central
  3, 5,  # West North Central - East North Central
  4, 6,  # West South Central - East South Central
  6, 7,  # East South Central - South Atlantic
  5, 8,  # East North Central - Mid-Atlantic
  7, 8,  # South Atlantic - Mid-Atlantic
  8, 9   # Mid-Atlantic - New England
), byrow = TRUE, ncol = 2)


# Create undirected graph
g <- graph_from_edgelist(edges, directed = FALSE)
adjacency_matrix <- as.matrix(as_adjacency_matrix(g))


# y <- ifelse(data$risk_level == 1, 1, -1)
# data2 <- data; colnames(data2) <- NULL
# X <- as.matrix(cbind(1,apply(data2[1:99], 2, as.numeric)))


params <- expand.grid(
  batch = 1:100, # Number of batches
  flip_prop = c(0, 0.01, 0.02) # Proportion of flipped labels
)

# === slurm_apply === #
sjob <- slurm_apply(
  f = real_one_batch,
  params = params,
  dat_crime = dat_crime, adjacency_matrix = adjacency_matrix, region_mapping = region_mapping,
  nodes = 100,
  jobname = paste(simulation_name, Sys.time()),
  global_objects = ls(),
  slurm_options = list(time = "0:30:00", `mem-per-cpu` = "500MB"))



# Save job object
saveRDS(sjob, paste0("Output/", "real_data", "_sjob.RDS"))
saveRDS(sjob, paste0("Output/", paste("real_data_", Sys.time()), "_sjob.RDS"))


# ====================================================== #
# Collect results
# ====================================================== #
sjob <- readRDS(paste0("Output/", "real_data", "_sjob.RDS"))
r <- get_slurm_out(sjob, "table", wait = TRUE)
colnames(r) <- c(
  "accuracy_desvm",
  "accuracy_pooled",
  "accuracy_init",
  "accuracy_subGD",
  "accuracy_DC",
  "nonzero_desvm",
  "nonzero_pooled",
  "nonzero_init",
  "nonzero_subGD",
  "nonzero_DC",
  "flip_prop"
)

aggregated_results <- r %>%
  group_by(flip_prop) %>%
  summarise(
    accuracy_desvm = mean(accuracy_desvm, na.rm = TRUE),
    accuracy_pooled = mean(accuracy_pooled, na.rm = TRUE),
    accuracy_init = mean(accuracy_init, na.rm = TRUE),
    accuracy_subGD = mean(accuracy_subGD, na.rm = TRUE),
    accuracy_DC = mean(accuracy_DC, na.rm = TRUE),
    nonzero_desvm = mean(nonzero_desvm, na.rm = TRUE),
    nonzero_pooled = mean(nonzero_pooled, na.rm = TRUE),
    nonzero_init = mean(nonzero_init, na.rm = TRUE),
    nonzero_subGD = mean(nonzero_subGD, na.rm = TRUE),
    nonzero_DC = mean(nonzero_DC, na.rm = TRUE)
  )
# Save aggregated results
aggregated_results <- as.data.frame(aggregated_results)
print(aggregated_results)

result_table <- xtable::xtable(aggregated_results, digits = 4)
result_table
print(result_table, include.rownames = FALSE, file = "Output/sim_real_data.tex")


saveRDS(r, file = paste0(
  "Output/LOG/sim_real_data",
  "_",
  format(Sys.time(), "%Y%m%d"))) #"%Y%m%d%H%M%S"
