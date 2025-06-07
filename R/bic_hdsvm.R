#' Parameter estimation using Bayesian Information Criterion (BIC) for high-dimensional SVM
#'
#' This function implements parameter selection using BIC for high-dimensional support vector machines,
#' handling both initial and pooled estimates.
#'
#' @param X Matrix of predictors (with intercept in first column)
#' @param y Vector of response values (-1, 1)
#' @param nlambda Number of lambda values for regularization path
#' @param h Tuning parameter for hinge loss (default: 0.05)
#' @param isinit Logical; if TRUE uses cross-validation, if FALSE uses BIC (default: FALSE)
#' @return Vector of coefficient estimates (intercept followed by predictors)
#' @export
bic.hdsvm <- function(X, y, nlambda = 100, h = 0.05, isinit = FALSE) {
  p <- ncol(X)
  N <- nrow(X)

  if (isinit) {
    cv.out <- hdsvm::cv.hdsvm(X[, -1], y, hval = h, nlambda = nlambda)
    out <- hdsvm::hdsvm(X[, -1], y, hval = h, nlambda = nlambda)
    ii <- which.min(cv.out$cvm)
    beta_pooled <- c(out$b0[ii], out$beta[, ii])
  } else {
    bic_array <- rep(NA, nlambda)
    out <- hdsvm::hdsvm(X[, -1], y, hval = h, nlambda = nlambda)

    for (ilambda in 1:ncol(out$beta)) {
      beta <- out$beta[, ilambda]
      beta0 <- out$b0[ilambda]
      bic_array[ilambda] <- sum(pmax(1 - y * (X[, -1] %*% beta + beta0), 0)) +
        log(N) *
          sqrt(log(N)) *
          sum(abs(beta) > 0)
    }
    ii <- which.min(bic_array)
    beta_pooled <- c(out$b0[ii], out$beta[, ii])
  }
  return(beta_pooled)
}
