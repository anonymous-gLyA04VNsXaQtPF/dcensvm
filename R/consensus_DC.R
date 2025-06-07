#' Compute the graph Laplacian matrix
#'
#' Calculates either the normalized or unnormalized graph Laplacian matrix from an adjacency matrix.
#'
#' @param adj_matrix A square symmetric adjacency matrix representing the graph
#' @param normalized Logical; if TRUE returns the normalized Laplacian, otherwise returns the unnormalized Laplacian (default: FALSE)
#' @return The graph Laplacian matrix
graph_laplacian <- function(adj_matrix, normalized = FALSE) {
  # Degree matrix
  degrees <- rowSums(adj_matrix)
  D <- diag(degrees)

  if (normalized) {
    # Compute the normalized Laplacian
    D_inv_sqrt <- diag(1 / sqrt(degrees))
    L_norm <- diag(nrow(adj_matrix)) - D_inv_sqrt %*% adj_matrix %*% D_inv_sqrt
    return(L_norm)
  } else {
    # Compute the un-normalized Laplacian
    L <- D - adj_matrix
    return(L)
  }
}

#' Performs decentralized parameter estimation via network consensus average
#'
#' Implements consensus algorithm from: http://nanodynamics.ece.umn.edu/data/e74747ab462302901dca.pdf
#' Uses graph Laplacian to achieve parameter consensus across network nodes.
#' Convergence guaranteed with step size epsilon < 1/max_degree.
#'
#' @param B Initial parameter matrix (p × m) with m nodes and p parameters.
#' @param adjacency_matrix Adjacency matrix defining network connectivity (m × m).
#' @param T Number of consensus iterations.
#' @return Consensus parameter matrix (p × m) after T communication steps.
#' @export
consensus_DC <- function(B, adjacency_matrix, T) {
  L <- graph_laplacian(adjacency_matrix)
  D <- rowSums(adjacency_matrix)
  eps <- 1/2/max(D)
  m <- nrow(adjacency_matrix)
  P <- diag(m) - eps*L
  for(it in 1:T) {
    B <- P%*%t(B)
    B <- t(B)
  }
  return(B)
}
