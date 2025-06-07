#include <RcppArmadillo.h>
#include <cmath>
#include "decentralizedsvm.h"

// [[Rcpp::depends(RcppArmadillo)]]


//' Implements subgradient descent for hinge loss SVM with fixed regularization.
//'
//' This function trains a linear SVM using the hinge loss with L1 regularization.
//' It uses a decaying step size schedule: eta = (a_lambda/(t+1))^{b_lambda}.
//' Convergence is monitored via relative coefficient change between iterations.
//'
//' @param X Design matrix of size N × p (N samples, p features).
//' @param y Response vector of length N (values +1 or -1).
//' @param b_init Initial coefficient vector of length p.
//' @param T Maximum number of iterations.
//' @param a_lambda Step size scaling parameter (default=3.5).
//' @param b_lambda Step size decay exponent (default=0.51).
//' @param lambda L1 regularization parameter (default=1e-2).
//' @param eps Convergence threshold for relative coefficient change (default=1e-4).
//' @param quiet If true, suppresses console output (default=true).
//' @return A list containing:
//'   - `b`: Final estimated coefficient vector (length p)
//'   - `history`: List containing:
//'        - `errors`: Vector of relative coefficient changes at each iteration
//' @export
// [[Rcpp::export]]
Rcpp::List SubGD_single_hinge(arma::mat &X,
                               arma::vec &y,
                               arma::mat &b_init,
                               int T = 20,
                               double a_lambda = 3.5, // for stepsize
                               double b_lambda = 0.51, // for stepsize
                               double lambda = 1e-2, // penalty parameter
                               double eps = 1e-4,
                               bool quiet = true) {

   int p = X.n_cols,
     N = X.n_rows;
   arma::vec g_mat(p); // subgradient matrix
   arma::vec b(p), b_old(p);
   arma::vec errors(T, arma::fill::zeros);
   double eta;


   b = b_init;
   int t, j;

   for(t = 0; t < T; t++) { // Iteration
     b_old = b;
     g_mat.zeros(); // Reset the gradient

     // Compute the gradient for hinge loss
     for (int i = 0; i < N; i++) {
       double margin = y(i) * arma::dot(X.row(i), b_old);
       if (margin < 1) { // hinge loss: only update if margin < 1
         g_mat -= y(i) * X.row(i).t(); // Gradient for hinge loss
       }
     }

     g_mat /= N; // Normalize by the number of samples
     g_mat += lambda * arma::sign(b_old); // Add regularization term

     // Update the coefficients using the subgradient
     eta = std::pow(a_lambda / (t + 1), b_lambda); // Learning rate
     b = b_old - eta * g_mat;

     // Compute the error between current b and true betaT
     errors(t) = std::sqrt(std::pow(arma::norm(b - b_old, "fro"), 2))/arma::norm(b_old, "fro");

     // Check for convergence
     if((t > 0) && (std::abs(errors(t) - errors(t - 1)) < eps)) {
       break;
     }
   }

   // Remove unused error values if stopped early
   if ((t + 1) <= (T - 1)) {
     errors.shed_rows(t + 1, T - 1);
   }

   return Rcpp::List::create(
     Rcpp::Named("b") = b,
     Rcpp::Named("history") = Rcpp::List::create(Rcpp::Named("errors") = errors));
 }

//' Implements SVM training with BIC-regularized parameter selection for hinge loss.
//'
//' This function selects optimal lambda via BIC criterion from a regularization path.
//' Lambda values are logarithmically spaced from lambda_max to lambda_max*lambda_factor.
//' For each lambda, trains a model using SubGD_single_hinge and computes BIC as:
//' BIC = hinge_loss + log(N)*sqrt(log(N))*shat, where shat is number of non-zero coefficients.
//' Final model uses lambda minimizing BIC.
//'
//' @param X Design matrix of size N × p (N samples, p features).
//' @param y Response vector of length N (values +1 or -1).
//' @param b_init Initial coefficient vector of length p.
//' @param T Maximum number of iterations per lambda.
//' @param a_lambda Step size scaling parameter (default=3.5).
//' @param b_lambda Step size decay exponent (default=0.51).
//' @param nlambda Number of lambda values in regularization path (default=100).
//' @param lambda_factor Maximum lambda in the regularization path before multiplied scale.
//' @param lambda_max Multiplied scale of all lambda in the regularization path.
//' @param eps Convergence threshold for relative coefficient change (default=1e-4).
//' @param quiet If true, suppresses console output (default=true).
//' @return A list containing (for optimal lambda):
//'   - `b`: Final estimated coefficient vector (length p)
//'   - `history`: List containing:
//'        - `errors`: Vector of relative coefficient changes at each iteration
//' @export
// [[Rcpp::export]]
Rcpp::List SubGD_hinge(arma::mat &X,
                        arma::vec &y,
                        arma::vec &b_init,
                        int T = 20,
                        double a_lambda = 3.5, // for stepsize
                        double b_lambda = 0.51, // for stepsize
                        int nlambda = 100L,
                        double lambda_factor = 1e-4,
                        double lambda_max = 1,
                        double eps = 1e-4,
                        bool quiet = true) {

   int N = X.n_rows;
   arma::vec b;
   double lambda;
   arma::vec bic_array(nlambda - 1);
   arma::vec lambda_array(nlambda);
   Rcpp::List L;


   lambda_array = arma::exp(arma::linspace(std::log(1), std::log(lambda_factor), nlambda));
   lambda_array.shed_row(0);
   lambda_array *= lambda_max;

   if (!quiet) {
     Rcpp::Rcout << "lambda_max = " << lambda_max << std::endl;
   }

   bic_array = arma::zeros(arma::size(lambda_array));
   arma::vec shat_array = arma::zeros(arma::size(lambda_array));

   // BIC computation for each lambda
   int shat;
   for (int ilambda = 0; ilambda < lambda_array.n_elem; ilambda++) {
     Rcpp::checkUserInterrupt(); // checking interruption
     lambda = lambda_array(ilambda);

     L = SubGD_single_hinge(X,
                            y,
                            b_init,
                            T,
                            a_lambda,
                            b_lambda,
                            lambda,
                            eps,
                            quiet);
     b = Rcpp::as<arma::vec>(L["b"]);

     // Compute BIC
     arma::vec result = arma::ones<arma::vec>(y.n_elem) - y % (X * b);
     bic_array(ilambda) += arma::accu(pmax_arma( result, 0.0));
     shat = std::max(shat, int(arma::as_scalar(arma::accu(b != 0))));
     bic_array(ilambda) = bic_array(ilambda) + std::log(N) * std::sqrt(std::log(N)) * std::max(shat, 1);
   }

   // Select best lambda based on BIC
   arma::uword ilambda = arma::index_min(bic_array);
   lambda = lambda_array(ilambda);

   if (!quiet) {
     Rcpp::Rcout << "lambda = " << lambda << std::endl;
   }

   // Run again with the best lambda
   L = SubGD_single_hinge(X,
                          y,
                          b_init,
                          T,
                          a_lambda,
                          b_lambda,
                          lambda,
                          eps,
                          quiet);
   return L;
 }
