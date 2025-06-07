#include <iostream>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

//' RcppArmadillo implementation of the R pmax()
//' function
//'
//' @title RcppArmadillo implemenation of the r pmax function
//' @param x the (aramdillo) vector
//' @param bound the (double) bound
//' @export
// [[Rcpp::export]]
arma::vec pmax_arma(arma::vec x, double bound) {
   int length_x = x.n_elem;
   for (int i = 0; i < length_x; i++) {
     if (x(i) < bound) x(i) = bound;
   }
   return x;
}


//' Computes a kernel function for density estimation.
//'
//' This kernel satisfies ∫K(u)du=1, ∫uK(u)du=0, and ∫u^2K(u)du=0.
//' The kernel is defined as: K(u) = [-315u^6 + 735u^4 - 525u^2 + 105]/64 for |u|<1, 0 otherwise.
//'
//' @param E Matrix of input values where kernel should be evaluated.
//' @param h Bandwidth vector controlling the kernel width.
//' @return Matrix of kernel density estimates scaled by bandwidth.
// [[Rcpp::export]]
arma::mat kernel(arma::mat E, arma::vec h) {
   arma::mat f(arma::size(E));
   E = E / arma::repmat(h.t(), E.n_rows, 1);
   // E.each_row() /= h.t();
   f = (-315 * arma::pow(E, 6) + 735 * arma::pow(E, 4) -
     525 * arma::pow(E, 2) + 105) /
       64 % (arma::abs(E) < 1);
   return arma::trans(arma::mean(f, 0)) / h;
}




double mod(arma::vec v) {
   std::map<int, size_t> frequencyCount;
   using pair_type = decltype(frequencyCount)::value_type;

   for (auto i : v)
     frequencyCount[i]++;

   auto pr = std::max_element
   (
       std::begin(frequencyCount), std::end(frequencyCount),
       [] (const pair_type & p1, const pair_type & p2) {
         return p1.second < p2.second;
       }
   );
   return pr->first;
 }



//' Generates consecutive indices for block j in a partitioned dataset.
//'
//' This function creates index sequences for data partitioned into equal-sized blocks.
//'
//' @param n Size of each data block (number of samples per partition).
//' @param j Block index (0-based).
//' @return Vector of consecutive indices for the j-th block: [n*j, n*(j+1)-1].
//' @export
// [[Rcpp::export]]
arma::uvec calN_j_cpp(int n, int j) {
   return arma::regspace<arma::uvec>(n * (j), 1, n * (j + 1) - 1);
 }


//' Applies soft thresholding operator for L1 regularization.
//'
//' This implements the proximal operator for L1 norm: S_t(x) = sign(x)*max(|x|-t, 0).
//'
//' @param x Input vector of coefficients.
//' @param t Threshold parameter (controls sparsity level).
//' @return Thresholded vector where values within [-t, t] are set to zero.
// [[Rcpp::export]]
arma::vec soft_thresholding_cpp(arma::vec x, double t) {
   return pmax_arma(x - t, 0) - pmax_arma(-x - t, 0);
}
