# Simulation: Sample Size under slurm jobs in HPC 
# Section K.9 Effect of the Sample Size $N$   
# Sample size effect analysis that we fix the number of nodes to $m = 10$ and the variable dimension to $p = 100$, and consider a fully connected communication network with connection probability $p_c = 1$. The local sample size $n \in \{100, 200, 400\}$ varies per node. 

rm(list = ls())
simulation_name <- "sample_size"
params_slurm <- expand.grid(
  batch = 1:100,
  n = c(100, 200, 400),
  pc = 1,
  rho = c(0.3, 0.5, 0.7, 0.9)
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


sjob <- slurm_apply(simulation, params_slurm, nodes = 100,
                    jobname = paste(simulation_name, Sys.time()),
                    global_objects = ls(),
                    slurm_options = list(time = "3:00:00", `mem-per-cpu` = "750MB"))


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
  select(starts_with("RMSE"), starts_with("F1_"),
         setdiff(names(params_slurm), c("batch", "pc")))


out_table <- aggregate(as.formula(paste0(". ~ ", paste(setdiff(names(params_slurm),  c("batch", "pc")), collapse = "+"))),
                       result, mean)
out_table
write.csv(out_table, paste0("Output/", simulation_name, "_result.csv"), row.names = F)
saveRDS(out_table, paste0("Output/", simulation_name, "_result.RDS"))

table_output <- xtable(out_table, digits = 4,
                       caption = paste0("Simulation results for ", simulation_name),
                       label = paste0("tab:sim_", simulation_name))
print(table_output, file = paste0("Output/", simulation_name, "_result.tex"), include.rownames = FALSE)

save.image(paste0("Output/", simulation_name, "_all_data.RData"))