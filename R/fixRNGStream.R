#' @title Fix Random Number Generator Stream
#' @description
#' This function sets up the random number generator (RNG) stream for reproducibility.
#' @param batch an integer scalar specifying the batch number for the RNG stream.
#' @param seed an integer scalar specifying the seed for the RNG.
#' @return The updated RNG stream.
#' @export
fixRNGStream <- function(batch = 1, seed = 42) {
  # Ensure inputs are valid
  stopifnot(is.numeric(batch), batch >= 0, is.numeric(seed))

  # Set the RNG kind and seed
  RNGkind("L'Ecuyer-CMRG")
  set.seed(seed)

  # Initialize the RNG stream
  s <- .Random.seed

  # Generate the next RNG stream for the specified batch
  for (i in seq_len(batch)) {
    s <- parallel::nextRNGStream(s)
  }

  # Optionally, update the global random seed (can be removed if not needed)
  .GlobalEnv$.Random.seed <- s

  # Return the updated seed
  return(s)
}
