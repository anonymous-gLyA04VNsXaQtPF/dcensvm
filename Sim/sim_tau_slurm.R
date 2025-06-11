# Simulation: Tuning parameter tau for Lagrangian penalty under slurm jobs in HPC 
# Section K.10 Augmented Parameter $tau$
# Evaluate the performance of deCSVM with $tau \in \{0.5, 1, 2, 4\}$.


rm(list = ls())
simulation_name <- "tau"
params_slurm <- expand.grid(
  batch = 1:100,
  tau_penalty_factor = c(0.5, 1, 2, 4),
  rho = c(0.3, 0.5, 0.7)
)

# ============================================================== #
# LOAD LIBRARY
# ============================================================== #
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
library(rslurm) # submit slurm jobs in HPC
library(peakRAM) #  to check memory
source("utils/sim_utils.R")

# peakRAM::peakRAM(simulation(p = 500, rho = 0.5, batch = 1))


sjob <- slurm_apply(simulation_iteration, params_slurm, nodes = 100,
                    jobname = paste(simulation_name, Sys.time()),
                    global_objects = ls(),
                    slurm_options = list(time = "2:00:00", `mem-per-cpu` = "500MB"))


# Save job object
saveRDS(sjob, paste0("Output/", simulation_name, "_sjob.RDS"))
saveRDS(sjob, paste0("Output/", paste(simulation_name, Sys.time()), "_sjob.RDS"))

# rslurm::get_job_status(sjob)$queue


# ====================================================== #
# Collect results
# ====================================================== #
library(dplyr)
library(xtable)

sjob <- readRDS(paste0("Output/", simulation_name, "_sjob.RDS"))
result <- get_slurm_out(sjob, "table", wait = T)
saveRDS(result, paste0("Output/", simulation_name, "_all_result.RDS"))

result <- result %>%
  dplyr::select(starts_with("RMSE"),
         setdiff(names(params_slurm), c("batch")))


out_table <- aggregate(as.formula(paste0(". ~ ", paste(setdiff(names(params_slurm), c("batch")), collapse =
  "+"))),
                       result, mean)
out_table
write.csv(out_table, paste0("Output/", simulation_name, "_result.csv"), row.names = F)
saveRDS(out_table, paste0("Output/", simulation_name, "_result.RDS"))

save.image(paste0("Output/", simulation_name, "_all_data.RData"))

tmp <- out_table %>%
    dplyr::select(rho, tau_penalty_factor, dplyr::last_col()) %>%
    tidyr::pivot_wider(names_from = tau_penalty_factor, values_from = dplyr::last_col())
tmp_table <- xtable::xtable(tmp, digits = 4)
print(tmp_table, include.rownames = FALSE, file = paste0("Output/", simulation_name, "_result_table.tex"))


# ====================================================== #
# Plot results
# ====================================================== #

plot_func <- function (result) {
  op <- par(no.readonly = TRUE)
fontsize <- 12
type <- "l"
lwd <- 2
par(
  font = fontsize,
  font.axis = fontsize,
  font.lab = fontsize,
  font.main = fontsize,
  font.sub = fontsize,
  cex = 1.5,
  mar = c(2,2,1,1)
)
plot(
  (colMeans(result[1,],na.rm = T)[1:501]),
  type = type,
  lwd = lwd,
  col = "black",#"firebrick",
  # ylim = c(0.2,0.9),
  xlab = "",
  ylab = ""
)
lines(
  (colMeans(result[2,],na.rm = T)[1:501]),#[202:401],#[1:201],
  type = type,
  pch = 2,
  col = "gold2",#"black", #"gold2",#adjustcolor("gold2", alpha.f = 0.1),
  lty = 2,
  lwd = lwd
)
lines(
  (colMeans(result[3, ],na.rm = T)[1:501]),#[202:401],#[1:201],
  type = type,
  pch = 5,
  col =  "firebrick",#"blue3",#"springgreen4",#"blue3",
  lty = "dotted",#5,
  lwd = lwd
)
lines(
  colMeans(result[4, ],na.rm = T)[1:501],#[202:401],#[1:201],
  type = type,
  pch = 4,
  col =  "blue3",#"turquoise3",
  lty = "longdash",#3,
  lwd = lwd
)
  par(op)
}


fig_dir <- "Output/figs"
for(cur_rho in unique(out_table$rho)) {
  tmp <- out_table %>%
    filter(rho == cur_rho) %>%
    dplyr::select(starts_with("RMSE"))

  pdf(paste0(fig_dir, "/fig_iterations", simulation_name, "_", cur_rho, ".pdf"))
  plot_func(tmp)
  dev.off()
}
