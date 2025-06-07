# Simulation: Dimension_increased under slurm jobs in HPC 
# Section K.4 Effect of Dimension  
# Assess the performance of our deCSVM method under increasing variable dimensionality with $p \in (100, 200, 300, 400, 500)$. 

rm(list = ls())
simulation_name <- "dimension_increased"


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
source("utils/sim_utils.R")
library(peakRAM) #  to check memory

# peakRAM::peakRAM(simulation(p = 500, rho = 0.5, batch = 1))

params_slurm <- expand.grid(
  batch = 1:100,
  p = c(100, 200, 300, 400, 500),
  rho = c(0.3, 0.5, 0.7, 0.9)
)

sjob <- slurm_apply(simulation, params_slurm, nodes = 100,
                    jobname = paste(simulation_name, Sys.time()),
                    global_objects = ls(),
                    slurm_options = list(time = "3:00:00", `mem-per-cpu` = "450MB"))



# Save job object
saveRDS(sjob, paste0("Output/", simulation_name, "_sjob.RDS"))
saveRDS(sjob, paste0("Output/", paste(simulation_name, Sys.time()), "_sjob.RDS"))

# rslurm::get_job_status(sjob)$queue



# ====================================================== #
# Collect results
# ====================================================== #
sjob <- readRDS(paste0("Output/", simulation_name, "_sjob.RDS"))
result <- get_slurm_out(sjob, "table", wait = T)
saveRDS(result, paste0("Output/", simulation_name, "_all_result.RDS"))

# library(tidyverse)
library(dplyr)

out_table <- aggregate(. ~ p + rho, result[, !(names(result) == "batch")], mean)
out_table
write.csv(out_table, paste0("Output/", simulation_name, "_result.csv"), row.names = F)
saveRDS(out_table, paste0("Output/", simulation_name, "_result.RDS"))
# library(readr)
library(xtable)
table_output <- xtable(out_table, digits = 4,
       caption = paste0("Simulation results for ", simulation_name),
       label = paste0("tab:sim_", simulation_name),
                       include.rownames = FALSE)
print(table_output, file = paste0("Output/", simulation_name, "_result.tex"))

save.image(paste0("Output/", simulation_name, "_all_data.RData"))
