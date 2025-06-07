#' Generate SVM Data with Erdos-Renyi Graph
#'
#' This function generates synthetic data for a Support Vector Machine (SVM) problem
#' with an Erdos-Renyi graph structure. It creates a feature matrix, response vector,
#' true coefficients, and a graph representation.
#'
#' @param N Total number of samples.
#' @param m Number of nodes in graph.
#' @param p Number of features (including intercept).
#' @param s Number of support (non-zero coefficients).
#' @param pc Edge probability for Erdos-Renyi graph generation.
#' @param mu Mean shift magnitude for support.
#' @param rho Autocorrelation parameter for Toeplitz covariance (default=0.1).
#' @param sigma2 Variance scaling factor (default=1).
#' @param ishomo Logical indicating whether to use homogeneous data generation (TRUE) or heterogeneous (FALSE). Currently only homogeneous generation is implemented. Default is TRUE.
#' @param hetercase Integer specifying the heterogeneity pattern to use when ishomo=FALSE. Reserved for future implementation. Default value is 1.
#' @param prop Proportion of labels to flip (label noise).
#' @return List containing:
#'   - `X`: Feature matrix (N × p)
#'   - `y`: Response vector (+1/-1) of length N
#'   - `betaT`: True Bayes-optimal coefficients (p × 1)
#'   - `graph`: Erdos-Renyi graph (igraph object) with m nodes
#' @export
gensvmData <- function(N, m, p, s, pc, mu, rho = .1, sigma2 = 1,  ishomo = TRUE,
                       hetercase = 1,
                       prop) {

  if (ishomo) {
    X <- matrix(NA, nrow = N, ncol = p)
    Y <- rbinom(N, 1, 0.5) * 2 -1;

    library(Matrix)
    # Generate homogenous data
    Sigma <- sigma2 * toeplitz(rho^seq(0, s - 1, by = 1))
    Sigma2 <- sigma2 * toeplitz(rho^seq(0, p-1-s - 1, by = 1))
    bSigma <-  as.matrix(bdiag(Sigma, Sigma2))
    ind_tmp <- (Y == 1)
    X[ind_tmp, ] <- cbind(1,MASS::mvrnorm(sum(ind_tmp), c(rep(mu,s),rep(0,p-1-s)), bSigma ))
    ind_tmp <- (Y == -1)
    X[ind_tmp, ] <- cbind(1,MASS::mvrnorm(sum(ind_tmp), c(rep(-mu,s),rep(0,p-1-s)), bSigma ))

    ## flip prop labels
    ind_tmp <- sample(1:N, floor(N*prop), replace = F)
    Y[ind_tmp] <- -1*Y[ind_tmp]
    s <- s



    # === find true betaT
    phi <- function(a) {
      return(dnorm(a))  # normal pdf
    }

    Phi <- function(a) {
      return(pnorm(a))  # normal cdf
    }

    gamma_a <- function(a) {
      return(phi(a) / Phi(a))
    }


    inverse_gamma_a <- function(target_value, lower = -10, upper = 10) {
      uniroot(function(a) gamma_a(a) - target_value, lower = lower, upper = upper)$root
    }

    # calculate dΣ(μ_f, μ_g)
    d_Sigma <- function(mu_f, mu_g, Sigma_inv) {
      diff <- mu_f - mu_g
      return(sqrt(t(diff) %*% Sigma_inv %*% diff))
    }

    # define function to calculate beta_0^*
    beta_0_star <- function(mu_f, mu_g, Sigma_inv, a_star, d_Sigma_value) {
      numerator <- t(mu_f - mu_g) %*% Sigma_inv %*% (mu_f + mu_g)
      denominator <- 2 * a_star * d_Sigma_value + d_Sigma_value^2
      return(-numerator / denominator)
    }

    # define function to calculate beta_+^*
    beta_plus_star <- function(mu_f, mu_g, Sigma_inv, a_star, d_Sigma_value) {
      numerator <- 2 * Sigma_inv %*% (mu_f - mu_g)
      denominator <- 2 * a_star * d_Sigma_value + d_Sigma_value^2
      return(numerator / as.numeric(denominator))
    }

    mu_f <- c(rep(mu, s), rep(0, p - s-1))
    mu_g <- c(rep(-mu, s), rep(0, p - s-1))

    # Calculate inverse matrix of covariance matrix
    Sigma_inv <- solve(bSigma)

    # Calculate d_Sigma and a_star
    d_Sigma_value <- d_Sigma(mu_f, mu_g, Sigma_inv)
    a_star <- inverse_gamma_a(d_Sigma_value / 2)

    # Calculate beta_0^* and beta_+^*
    beta_0 <- beta_0_star(mu_f, mu_g, Sigma_inv, a_star, d_Sigma_value)
    beta_plus <- beta_plus_star(mu_f, mu_g, Sigma_inv, a_star, d_Sigma_value)


    betaT <- c(beta_0, beta_plus)

  }
  graph <- igraph::sample_gnp(m, pc)
  while(!igraph::is_connected(graph)) {
    graph <- igraph::sample_gnp(m, pc)
  }

  return(list(
    X = X,
    y = Y,
    betaT = betaT,
    graph = graph
  ))
}




#' @title Create Directory if Not Exists
#' @description Utility function to safely create directories. Equivalent to mkdir -p in Unix.
#' @param path Directory path to create.
#' @param recursive Create parent directories if needed (default=TRUE).
#' @return Logical: TRUE if directory was created, FALSE if already existed.
#' @export
createdir <- function(path, recursive = T) {
  ifelse(!dir.exists(path), dir.create(path, recursive = recursive), FALSE)
}



#' @title Combine Elements from Multiple Lists
#' @description Utility function for use with foreach to combine elements from multiple lists by position
#' @param x First list to combine
#' @param ... Additional lists to combine with the first list
#' @return A list where each element contains the corresponding elements from all input lists
#' @export
comb <- function(x, ...) {
  lapply(seq_along(x),
         function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}







#' @title Compute F1 score of two set
#' @description Compute \eqn{F_1} score of two set. Note that \eqn{F_1} score is
#' symmetric.
#' @name computeF1
#' @param esupp the estimated support
#' @param supp the true support
#' @return A list consists of
#' \item{f1}{the computed F1 score}
#' \item{precision}{the precision}
#' \item{recall}{the recall}
#' \item{fp}{false positve}
#' \item{fn}{false negative}
#' @references
#' \url{https://en.wikipedia.org/wiki/F-score}
#' @export
computeF1 <- function(esupp, supp) {
  tp <- length(intersect(esupp, supp))
  fp <- length(setdiff(esupp, supp))
  fn <- length(setdiff(supp, esupp))
  precision <- tp/(tp + fp)
  recall <- tp/(tp + fn)
  f1 <- 2*(precision*recall)/(precision+recall)
  if(precision == 0 || recall == 0) {
    f1 <- 0
  }
  return(list(f1 = f1,
              precision = precision,
              recall = recall,
              fp = fp,
              fn = fn))
}
