# Simulation: BIC under slurm jobs in HPC 
# Section K.7 Effect of the Tuning Parameter $\lambda$
# Compares error and estimated sparsity level under different kernel types (kernel_type 1-5) using BIC criterion with different $\lambda$ in Figure K.3.

rm(list = ls())
simulation_name <- "BIC"

library(dplyr)
params_slurm <- expand.grid(
  batch = 1:100,
  kernel_type = 1:5
) %>%
  dplyr::select(batch, kernel_type)

# ============================================================== #
# LOAD LIBRARY
# ============================================================== #
library(future)
library(future.apply) # parallel package
library(doRNG)
library(foreach)
library(doFuture)
library(parallel)
library(tictoc)
library(MASS)
library(pracma)
library(igraph) # for graph
library(glmnet)
# library(ggplot2)
library(dcensvm)
library(hdsvm) # initial
library(peakRAM) #  to check memory
library(rslurm) # submit slurm jobs in HPC
source("utils/sim_utils.R")

sjob <- slurm_apply(simulation_lambda, params_slurm, nodes = 100,
                    jobname = paste(simulation_name, Sys.time()),
                    global_objects = ls(),
                    slurm_options = list(time = "2:00:00", `mem-per-cpu` = "500MB"))
# Save job object
saveRDS(sjob, paste0("Output/", simulation_name, "_sjob.RDS"))
saveRDS(sjob, paste0("Output/", paste(simulation_name, Sys.time()), "_sjob.RDS"))


# ====================================================== #
# Collect results
# ====================================================== #
library(dplyr)
library(xtable)

sjob <- readRDS(paste0("Output/", simulation_name, "_sjob.RDS"))

result <- get_slurm_out(sjob, "raw", wait = TRUE)
param_list <- split(params_slurm, seq_len(nrow(params_slurm)))


result_df <- do.call(rbind, lapply(seq_along(result), function(i) {
  run <- result[[i]]
  param <- param_list[[i]]

  #  data.frame
  run_df <- as.data.frame(do.call(rbind, run))[1:200, ] # nlambda = 200!
  colnames(run_df) <- c("lambda", "error", "rank", "BIC")

  # parameter into run_df
  param_df <- as.data.frame(param)[rep(1, nrow(run_df)), ]
  cbind(run_df, param_df)
}))
#result <- get_slurm_out(sjob, "table", wait = T)
# result <- cbind(params_slurm, result_df)
saveRDS(result_df, paste0("Output/", simulation_name, "_all_result.RDS"))



library(dplyr)

agg_df <- result_df %>%
  group_by(kernel_type, lambda) %>%
  summarise(
    error = mean(error, na.rm = TRUE),
    rank = mean(rank, na.rm = TRUE),
    BIC = mean(BIC, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(agg_df, paste0("Output/", simulation_name, "_result.csv"), row.names = F)
saveRDS(agg_df, paste0("Output/", simulation_name, "_result.RDS"))

save.image(paste0("Output/", simulation_name, "_all_data.RData"))




# ============================================================== #
# PLOT
# ============================================================== #


library(ggplot2)
agg_df <- readRDS(paste0("Output/", simulation_name, "_result.RDS"))

agg_df$kernel_type <- factor(agg_df$kernel_type)
levels(agg_df$kernel_type) <-  c("Uniform", "Laplacian", "Logistic", "Gaussian", "Epanechnikov")


metrics <- c("error", "rank", "BIC")
set1_colors <- c("black","gold2","firebrick","blue3","springgreen4")
for (metric in metrics) {
  p <- ggplot(agg_df, aes(x = lambda, y = .data[[metric]], color = kernel_type, linetype = kernel_type)) +
    geom_line(size = 0.7) +
    labs(x = "", y = "") +
    scale_color_manual(values = set1_colors) +
    theme_minimal(base_size = 14) +
    ggpubr::theme_pubr() +
    theme(
      legend.position = c(0.75, 1),  # Top-right corner inside the plot
      legend.justification = c( "top"),
      legend.title = element_blank(),
      legend.box.just = "right",
      legend.text = element_text(size = 12),
      legend.key.width = unit(2.5, "lines")  # Increase legend line length
    ) +
    guides(
      color = guide_legend(ncol = 1),
      linetype = guide_legend(ncol = 1)
    ) +
    scale_x_log10()  # since lambda is log

  ggsave(filename = paste0("Output/plot_lambda_vs_", metric, ".pdf"), plot = p, width = 8, height = 5)
}


# ============================================================== #
# Optimal
# ============================================================== #

best_lambda_per_kernel <- agg_df %>%
  group_by(kernel_type) %>%
  slice_min(BIC, n = 1, with_ties = FALSE) %>%
  ungroup()

print(best_lambda_per_kernel)


write.csv(best_lambda_per_kernel, "Output/best_lambda_by_kernel.csv", row.names = FALSE)

