# Simulation: Tuning parameter C_rho for quadratic majority under slurm jobs in HPC 
# Section K.6 Effect of the Tuning Parameter $\rho_\ell$
# Sensitivity analysis for $\rho_\ell = (1 + \delta)c_h \Lambda_{\max}( \frac{1}{n} \sum_{i \in \mathcal{I}_\ell} x_i
# x_i^\top)$ with $\delta \in \{0.01, 1, 4\}$ in Figure K.2.

rm(list = ls())
simulation_name <- "C_rho"
params_slurm <- expand.grid(
  batch = 1:100,
  C_rho = c(1.01, 2, 5),
  rho = c(0.3, 0.5)
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
  # ylim = c(0.2,0.9),#c(0, 1),
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
  par(op)
}


fig_dir <- "Output/figs"
for(cur_rho in unique(out_table$rho)) {
  tmp <- out_table %>%
    filter(rho == cur_rho) %>%
    dplyr::select(starts_with("RMSE"))

  pdf(paste0(fig_dir, "/fig_iterations_C_rho_", cur_rho, ".pdf"))
  plot_func(tmp)
  dev.off()
}
