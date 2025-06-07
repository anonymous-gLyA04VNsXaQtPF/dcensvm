# Function to compute metrics for different methods
compute_metrics <- function(X, y, B, m, n, p, betaT, suppT) {
  N <- nrow(X)
  loss_error <- matrix(rep(0, m), nrow = m)
  rmse <- 0

  for (j in 1:m) {
    idx <- ((j - 1) * n + 1):(j * n)
    loss_error[j] <- sum(pmax(1 - y[idx] * (X[idx,] %*% B[, j]), 0))
    rmse <- rmse + mean((B[, j] - betaT)^2)
  }

  error <- sum(loss_error) / m / n
  f1_score <- mean(unlist(lapply(apply(B, 2, function(x) computeF1(which(abs(x) > 0), suppT)), `[[`, 1)))
  # print(B)
  # print("suppT = ", suppT)
  list(
    error = error,
    f1_score = f1_score,
    rmse = sqrt(sum((B - repmat(matrix(betaT, p, 1), 1, m))^2) / m)
  )
}


# ============================================================== #
# Main function to run all methods
# ============================================================== #
run_all_methods <- function(data_list) {
  # Extract parameters from data_list
  X <- data_list$X
  y <- data_list$y
  adjacency_matrix <- data_list$adjacency_matrix
  betaT <- data_list$betaT
  suppT <- data_list$suppT
  params <- data_list$params

  RMSE <- rep(NA, 5)

  # ============================================================== #
  # Initial estimation
  # ============================================================== #
  B_init <- matrix(rep(0, params$m * params$p), nrow = params$p)

  for (j in 1:params$m) {
    idx <- ((j - 1) * params$n + 1):(j * params$n)
    idx_ind <- ((j - 1) * 50 * params$n + 1):(j * 50 * params$n)
    out <- dcensvm::bic.hdsvm(X[idx,], y[idx], nlambda = 100, isinit = T)
    B_init[, j] <- out
  }

  # Compute initial metrics
  metrics_init <- compute_metrics(X, y, B_init, params$m, params$n, params$p,
                                  betaT, suppT)
  error_init <- metrics_init$error
  f1_init <- metrics_init$f1_score
  RMSE[1] <- metrics_init$rmse

  # ============================================================== #
  # Decentralized kernel SVM
  # ============================================================== #
  # B_init_null <- matrix(0, params$p, params$m)
  lambda_max <- max(abs(t(X[, -1]) %*% y) / params$N)
  out_beta_desvm <- dcensvm::decentralizedsvm_cpp(
    X = X,
    y = y,
    adjacency_matrix = adjacency_matrix,
    B_init = B_init,
    betaT = betaT,
    # h = 0.01,
    h = params$h,
    type = params$kernel_type,
    C_N = sqrt(log(params$N)),
    C_rho = params$C_rho,
    T_outer = params$T_buget,
    c0 = params$c0,
    tau_penalty_factor = params$tau_penalty_factor,
    lambda_factor = params$lambda_factor,
    nlambda = params$nlambda,
    # lambda_max = lambda_max,
    lambda_max = params$lambda_max,
    lambda_0 = 0,
    quiet = params$quiet,
    eps = 1e-5
  )

  # Plot results
  # plot(out_beta_desvm$history$errors_outer)
  # plot(out_beta_desvm$history$loss_error)

  # Compute deSVM metrics
  metrics_desvm <- compute_metrics(X, y, out_beta_desvm$B, params$m, params$n, params$p,
                                   betaT, suppT)
  error_desvm <- metrics_desvm$error
  f1_desvm <- metrics_desvm$f1_score
  RMSE[2] <- metrics_desvm$rmse

  # ============================================================== #
  # DC method
  # ============================================================== #
  B_DC_tmp <- consensus_DC(B_init, adjacency_matrix, params$T_buget)

  # Compute DC metrics
  metrics_dc <- compute_metrics(X, y, B_DC_tmp, params$m, params$n, params$p,
                                betaT, suppT)
  error_DC <- metrics_dc$error
  f1_DC <- metrics_dc$f1_score
  RMSE[3] <- metrics_dc$rmse

  # ============================================================== #
  # Pooled method
  # ============================================================== #
  out_pooled_f <- matrix(dcensvm::bic.hdsvm(X, y, nlambda = 100, isinit = FALSE))
  B_pooled <- out_pooled_f

  # Compute pooled metrics
  metrics_pooled <- compute_metrics(X, y, matrix(rep(B_pooled, params$m), ncol = params$m), params$m, params$n, params$p,
                                    betaT, suppT)
  error_pooled <- metrics_pooled$error
  f1_pooled <- metrics_pooled$f1_score
  RMSE[4] <- metrics_pooled$rmse

  # ============================================================== #
  # deSubGD method
  # ============================================================== #
  out_deSubGD <- deSubGD_svm(
    X, y, adjacency_matrix,
    matrix(rnorm(params$p * params$m), params$p, params$m),
    T = params$T_buget, lambda_max = 20, eps = 1e-8, quiet = params$quiet
  )

  # Compute deSubGD metrics
  metrics_deSubGD <- compute_metrics(X, y, out_deSubGD$B, params$m, params$n, params$p,
                                     betaT, suppT)
  error_deSubGD <- metrics_deSubGD$error
  f1_deSubGD <- metrics_deSubGD$f1_score
  RMSE[5] <- metrics_deSubGD$rmse

  # ============================================================== #
  # Results summary
  # ============================================================== #
  tmp <- matrix(c(
    RMSE, error_init, error_desvm, error_DC, error_pooled, error_deSubGD,
    f1_init, f1_desvm, f1_DC, f1_pooled, f1_deSubGD
  ), 5, 3)

  rownames(tmp) <- c("Local", "Our", "DC", "Pooled", "deSubG")
  colnames(tmp) <- c("RMSE", "Objective value", "f1")
  tmp


  return(list(results = tmp, desvm_output = out_beta_desvm))
}


run_all_methods_iteration <- function(data_list) {
  # Extract parameters from data_list
  X <- data_list$X
  y <- data_list$y
  adjacency_matrix <- data_list$adjacency_matrix
  betaT <- data_list$betaT
  suppT <- data_list$suppT
  params <- data_list$params

  RMSE <- rep(NA, 5)

  # ============================================================== #
  # Initial estimation
  # ============================================================== #
  B_init <- matrix(rep(0, params$m * params$p), nrow = params$p)

  for (j in 1:params$m) {
    idx <- ((j - 1) * params$n + 1):(j * params$n)
    idx_ind <- ((j - 1) * 50 * params$n + 1):(j * 50 * params$n)
    out <- dcensvm::bic.hdsvm(X[idx,], y[idx], nlambda = 100, isinit = T)
    B_init[, j] <- out
  }

  # Compute initial metrics
  metrics_init <- compute_metrics(X, y, B_init, params$m, params$n, params$p,
                                  betaT, suppT)
  error_init <- metrics_init$error
  f1_init <- metrics_init$f1_score
  RMSE[1] <- metrics_init$rmse

  # ============================================================== #
  # Decentralized kernel SVM
  # ============================================================== #
  # B_init_null <- matrix(0, params$p, params$m)
  lambda_max <- max(abs(t(X[, -1]) %*% y) / params$N)
  out_beta_desvm <- dcensvm::decentralizedsvm_cpp(
    X = X,
    y = y,
    adjacency_matrix = adjacency_matrix,
    B_init = B_init,
    betaT = betaT,
    # h = 0.01,
    h = params$h,
    type = params$kernel_type,
    C_N = sqrt(log(params$N)),
    C_rho = params$C_rho,
    T_outer = params$T_buget,
    c0 = params$c0,
    tau_penalty_factor = params$tau_penalty_factor,
    lambda_factor = params$lambda_factor,
    nlambda = params$nlambda,
    # lambda_max = lambda_max,
    lambda_max = params$lambda_max,
    lambda_0 = 0,
    quiet = params$quiet,
    eps = 1e-5
  )

  # Plot results
  # plot(out_beta_desvm$history$errors_outer)
  # plot(out_beta_desvm$history$loss_error)



  return(out_beta_desvm$history$errors_outer)
}




run_all_methods_lambda <- function(data_list) {
  # Extract parameters from data_list
  X <- data_list$X
  y <- data_list$y
  adjacency_matrix <- data_list$adjacency_matrix
  betaT <- data_list$betaT
  suppT <- data_list$suppT
  params <- data_list$params

  RMSE <- rep(NA, 5)

  # ============================================================== #
  # Initial estimation
  # ============================================================== #
  B_init <- matrix(rep(0, params$m * params$p), nrow = params$p)

  for (j in 1:params$m) {
    idx <- ((j - 1) * params$n + 1):(j * params$n)
    idx_ind <- ((j - 1) * 50 * params$n + 1):(j * 50 * params$n)
    out <- dcensvm::bic.hdsvm(X[idx,], y[idx], nlambda = 100, isinit = T)
    B_init[, j] <- out
  }

  # Compute initial metrics
  metrics_init <- compute_metrics(X, y, B_init, params$m, params$n, params$p,
                                  betaT, suppT)
  error_init <- metrics_init$error
  f1_init <- metrics_init$f1_score
  RMSE[1] <- metrics_init$rmse

lambda_array <- params$lambda_array
    output_list <- vector("list", length(lambda_array))

for (iT_lambda in seq_along(lambda_array)) {
  lambda_i <- lambda_array[iT_lambda]

  out_beta_desvm <- dcensvm::decentralizedsvm_lambda(
    X = X,
    y = y,
    adjacency_matrix = adjacency_matrix,
    B_init = B_init,
    betaT = betaT,
    h = params$h,
    type = params$kernel_type,
    C_N = sqrt(log(params$N)),
    C_rho = params$C_rho,
    T_outer = params$T_buget,
    c0 = params$c0,
    tau_penalty_factor = params$tau_penalty_factor,
    lambda = lambda_i,
    lambda_0 = 0,
    quiet = params$quiet,
    eps = 1e-5
  )

    error <- out_beta_desvm$history$errors_outer[length(out_beta_desvm$history$errors_outer)]
    rank <- out_beta_desvm$history$shat_array[length(out_beta_desvm$history$shat_array)]
    BIC <- out_beta_desvm$history$loss_error[length(out_beta_desvm$history$loss_error)]

    output_list[[iT_lambda]] <- c(lambda_i, error, rank, BIC)
}

  return(output_list)
}


# ============================================================== #
# Function to run the simulation
# ============================================================== #
simulation <- function(batch = 1, s = 10, rho = 0.5, prop = 0.01, n = 200, m = 10, p = 100,
                       sigma2 = 1, mu = 0.4, pc = 0.5, lambda_factor = 1e-4,
                       lambda_max = 0.3, nlambda = 100L, quiet = FALSE, c0 = 0.013,
                       kernel_type = 3, T_buget = 200, tau_penalty_factor = 1,
                       C_rho = 1.01, C_h = 1) {
  fixRNGStream(batch)
  # set.seed(batch)
  if(!quiet) {
      cat("batch =", batch, "rho =", rho, "prop =", prop, "n =", n, "m =", m,
      "p =", p, "sigma2 =", sigma2, "mu =", mu, "pc =", pc, "lambda_factor =", lambda_factor,
      "lambda_max =", lambda_max, "nlambda =", nlambda, "quiet =", quiet, "c0 =", c0,
      "kernel_type =", kernel_type, "T_buget =", T_buget, "tau_penalty_factor =",
      tau_penalty_factor, "C_rho =", C_rho, "C_h = ", C_h, "s =", s, "\n")
  }

  params <- list(
    n = n,  # samples per machine
    m = m,
    sigma2 = sigma2,
    s = s,
    mu = mu,
    pc = pc,
    lambda_factor = lambda_factor,
    lambda_max = lambda_max,
    nlambda = nlambda,
    quiet = quiet,
    c0 = c0,
    kernel_type = kernel_type,
    T_buget = T_buget,
    rho = rho,
    p = p,
    tau_penalty_factor = tau_penalty_factor,
    C_rho = C_rho,
    C_h = C_h,
    prop = prop
  )
  params$N <- params$m * params$n
  params$h <- max(0.05, C_h*(log(params$p) / params$N)^(1 / 4))

  # Generate data
  data <- dcensvm::gensvmData(N = params$N,
                              m = params$m,
                              p = params$p,
                              s = params$s,
                              pc = params$pc,
                              mu = params$mu,
                              rho = params$rho,
                              sigma2 = params$sigma2,
                              ishomo = TRUE,
                              hetercase = 1,
                              prop = params$prop
  )

  betaT <- data$betaT
  X <- data$X
  y <- data$y
  graph <- data$graph
  adjacency_matrix <- as.matrix(as_adjacency_matrix(graph))


  suppT <- 1:params$s
  # Build data_list
  data_list <- list(
    X = X,
    y = y,
    adjacency_matrix = adjacency_matrix,
    betaT = betaT,
    suppT = suppT,
    params = params
  )

  # Run all methods
  results <- run_all_methods(data_list)

  # Save to output list
  out_tmp <- c(
    setNames(results$results[, "RMSE"], paste0("RMSE_", rownames(results$results))),
    setNames(results$results[, "Objective value"], paste0("Obj_", rownames(results$results))),
    setNames(results$results[, "f1"], paste0("F1_", rownames(results$results)))
  )
  # add names by RMSE, Objective value, accuracy, f1

  out_tmp <- c(out_tmp, batch = batch, rho = params$rho, s = params$s, n = params$n, m = params$m, p = params$p,
               sigma2 = params$sigma2, mu = params$mu, pc = params$pc,
               prop = params$prop, lambda_factor = params$lambda_factor,
               lambda_max = params$lambda_max, nlambda = params$nlambda,
               c0 = params$c0, kernel_type = params$kernel_type,
               T_buget = params$T_buget, tau_penalty_factor = params$tau_penalty_factor,
               C_rho = params$C_rho, C_h = params$C_h)

  return(out_tmp)
}



simulation_iteration <- function(batch = 1, s = 10, rho = 0.5, prop = 0.01, n = 200, m = 10, p = 100,
                       sigma2 = 1, mu = 0.4, pc = 0.5, lambda_factor = 1e-4,
                       lambda_max = 0.3, nlambda = 100L, quiet = FALSE, c0 = 0.013,
                       kernel_type = 3, T_buget = 500, tau_penalty_factor = 1,
                       C_rho = 1.01, C_h = 1) {
  fixRNGStream(batch)
  # set.seed(batch)

  if(!quiet) {
    cat("batch =", batch, "rho =", rho, "prop =", prop, "n =", n, "m =", m,
      "p =", p, "sigma2 =", sigma2, "mu =", mu, "pc =", pc, "lambda_factor =", lambda_factor,
      "lambda_max =", lambda_max, "nlambda =", nlambda, "quiet =", quiet, "c0 =", c0,
      "kernel_type =", kernel_type, "T_buget =", T_buget, "tau_penalty_factor =",
      tau_penalty_factor, "C_rho =", C_rho, "C_h = ", C_h, "s =", s, "\n")
  }

  params <- list(
    n = n,  # samples per machine
    m = m,
    sigma2 = sigma2,
    s = s,
    mu = mu,
    pc = pc,
    lambda_factor = lambda_factor,
    lambda_max = lambda_max,
    nlambda = nlambda,
    quiet = quiet,
    c0 = c0,
    kernel_type = kernel_type,
    T_buget = T_buget,
    rho = rho,
    p = p,
    tau_penalty_factor = tau_penalty_factor,
    C_rho = C_rho,
    C_h = C_h,
    prop = prop
  )
  params$N <- params$m * params$n
  params$h <- max(0.05, C_h*(log(params$p) / params$N)^(1 / 4))

  # Generate data
  data <- dcensvm::gensvmData(N = params$N,
                              m = params$m,
                              p = params$p,
                              s = params$s,
                              pc = params$pc,
                              mu = params$mu,
                              rho = params$rho,
                              sigma2 = params$sigma2,
                              ishomo = TRUE,
                              hetercase = 1,
                              prop = params$prop
  )

  betaT <- data$betaT
  X <- data$X
  y <- data$y
  graph <- data$graph
  adjacency_matrix <- as.matrix(as_adjacency_matrix(graph))


  suppT <- 1:params$s
  # Build data_list
  data_list <- list(
    X = X,
    y = y,
    adjacency_matrix = adjacency_matrix,
    betaT = betaT,
    suppT = suppT,
    params = params
  )

  # Run all methods
  results <- run_all_methods_iteration(data_list)


  # Save to output list
  out_tmp <- setNames(results, paste0("RMSE_", 1:length(results)))
  # add names by RMSE, Objective value, accuracy, f1

  out_tmp <- c(out_tmp, batch = batch, rho = params$rho, s = params$s, n = params$n, m = params$m, p = params$p,
               sigma2 = params$sigma2, mu = params$mu, pc = params$pc,
               prop = params$prop, lambda_factor = params$lambda_factor,
               lambda_max = params$lambda_max, nlambda = params$nlambda,
               c0 = params$c0, kernel_type = params$kernel_type,
               T_buget = params$T_buget, tau_penalty_factor = params$tau_penalty_factor,
               C_rho = params$C_rho, C_h = params$C_h)

  return(out_tmp)
}






simulation_lambda <- function(batch = 1, s = 10, rho = 0.5, prop = 0.01, n = 200, m = 10, p = 100,
                       sigma2 = 1, mu = 0.4, pc = 0.5, lambda_factor = 0.3, lambda_min = 1e-4,
                       lambda_max = 1, nlambda = 200L, quiet = FALSE, c0 = 0.013,
                       kernel_type = 3, T_buget = 200, tau_penalty_factor = 1,
                       C_rho = 1.01, C_h = 1) {
  fixRNGStream(batch)
  # set.seed(batch)

  if(!quiet) {
    cat("batch =", batch, "rho =", rho, "prop =", prop, "n =", n, "m =", m,
      "p =", p, "sigma2 =", sigma2, "mu =", mu, "pc =", pc, "lambda_factor =", lambda_factor, "lambda_min =", lambda_min,
      "lambda_max =", lambda_max, "nlambda =", nlambda, "quiet =", quiet, "c0 =", c0,
      "kernel_type =", kernel_type, "T_buget =", T_buget, "tau_penalty_factor =",
      tau_penalty_factor, "C_rho =", C_rho, "C_h = ", C_h, "s =", s, "\n")
  }

  exponent <- seq(log(lambda_min), log(lambda_max), length.out = nlambda)

# Calculate the corresponding lambda value
  lambda_array <- exp(exponent) * lambda_factor

  params <- list(
    n = n,  # samples per machine
    m = m,
    sigma2 = sigma2,
    s = s,
    mu = mu,
    pc = pc,
    lambda_factor = lambda_factor,
    lambda_min = lambda_min,
    lambda_max = lambda_max,
    nlambda = nlambda,
    quiet = quiet,
    c0 = c0,
    kernel_type = kernel_type,
    T_buget = T_buget,
    rho = rho,
    p = p,
    tau_penalty_factor = tau_penalty_factor,
    C_rho = C_rho,
    C_h = C_h,
    prop = prop
  )
  params$N <- params$m * params$n
  params$h <- max(0.05, C_h*(log(params$p) / params$N)^(1 / 4))
  params$lambda_array <- lambda_array

# Using the seq function to generate exponential power values for logarithmic grids


  # Generate data
  data <- dcensvm::gensvmData(N = params$N,
                              m = params$m,
                              p = params$p,
                              s = params$s,
                              pc = params$pc,
                              mu = params$mu,
                              rho = params$rho,
                              sigma2 = params$sigma2,
                              ishomo = TRUE,
                              hetercase = 1,
                              prop = params$prop
  )

  betaT <- data$betaT
  X <- data$X
  y <- data$y
  graph <- data$graph
  adjacency_matrix <- as.matrix(as_adjacency_matrix(graph))


  suppT <- 1:params$s
  # Build data_list
  data_list <- list(
    X = X,
    y = y,
    adjacency_matrix = adjacency_matrix,
    betaT = betaT,
    suppT = suppT,
    params = params
  )

  # Run all methods
  results <- run_all_methods_lambda(data_list)


  # # Save to output list
  # out_tmp <- setNames(results, paste0("RMSE_", 1:length(results)))
  # add names by RMSE, Objective value, accuracy, f1

  out_tmp <- c(results, batch = batch, rho = params$rho, s = params$s, n = params$n, m = params$m, p = params$p,
               sigma2 = params$sigma2, mu = params$mu, pc = params$pc,
               prop = params$prop, lambda_factor = params$lambda_factor,
               lambda_max = params$lambda_max, nlambda = params$nlambda,
               c0 = params$c0, kernel_type = params$kernel_type,
               T_buget = params$T_buget, tau_penalty_factor = params$tau_penalty_factor,
               C_rho = params$C_rho, C_h = params$C_h)

  return(out_tmp)
}

