#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "decentralizedsvm.h"
#include <cmath>


//' Implements decentralized subgradient descent (D-subGD) for SVM with fixed regularization parameter.
//'
//' This function performs decentralized training across multiple nodes defined by an adjacency matrix.
//' It uses a subgradient method with decaying step size (eta = a/(t+1)^b) and L1 regularization.
//' The algorithm converges when relative RMSE change falls below threshold.
//'
//' @param X Design matrix of size N × p (N total samples, p features).
//' @param y Response vector of length N (values should be +1 or -1).
//' @param adjacency_matrix Matrix defining communication links between nodes (m × m).
//' @param B_init Initial coefficient matrix of size p × m (one coefficient vector per node).
//' @param T Maximum number of iterations.
//' @param a Stepsize scaling parameter (default=3.5).
//' @param b Stepsize decay exponent (default=0.51).
//' @param lambda L1 regularization parameter (default=1e-2).
//' @param eps Convergence threshold for relative Frobenius norm change (default=1e-2).
//' @param quiet If true, suppresses console output (default=true).
//' @return A list containing:
//'   - `B`: Final estimated coefficient matrix (p × m)
//'   - `history`: List containing:
//'        - `errors`: Vector of relative RMSE at each iteration.
//' @export
// [[Rcpp::export]]
Rcpp::List deSubGD_single_svm(arma::mat &X,
                               arma::vec &y,
                               arma::mat &adjacency_matrix,
                               arma::mat &B_init,
                               int T = 20,
                               double a = 3.5, // for stepsize
                               double b = 0.51, // for stepsize
                               double lambda = 1e-2, // penalty parameter
                               double eps = 1e-2,
                               bool quiet = true) {

   int m = adjacency_matrix.n_cols,
     p = X.n_cols,
     N = X.n_rows,
     n = N / m;

   arma::mat g_mat(p, m); // subgradient matrix
   arma::mat B(size(B_init)), B_old(size(B_init));
   arma::mat Phi(size(B_init));

   arma::vec errors(T, arma::fill::zeros);
   arma::uvec idx(n);
   double eta;

   arma::mat C(m, m, arma::fill::zeros);
   arma::uvec n_arr(m, arma::fill::zeros);


   for(int j = 0; j < m; j++) {
     arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
     n_arr(j) = neighbors_j.n_elem;
   }
   for(int j = 0; j < m; j++) {
     arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
     for(int l = 0; l<neighbors_j.n_elem; l++) {
       C(neighbors_j(l),j) = 1.0/std::max(n_arr(j), n_arr(neighbors_j(l)));
     }
   }
   for(int j = 0; j < m; j++) {
     arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
     arma::vec tmp_C = C.col(j);
     C(j,j) = 1 - arma::sum(tmp_C(neighbors_j));
   }


   B = B_init;
   int t, j;
   for(t = 0; t < T; t++) {
     // Calculate subgradient
     for(j = 0; j < m; j++) {
       idx = calN_j_cpp(n, j);
       g_mat.col(j).zeros();  // initialize subgradient
       for (int i = 0; i < n; i++) {
         double margin = y(idx(i)) * arma::dot(X.row(idx(i)), B.col(j)); // y_i * (X_i * B_j)
         if (margin < 1) {
           g_mat.col(j) -= y(idx(i)) * X.row(idx(i)).t(); // update subgradient
         }
       }

       g_mat.col(j) /= N;
       g_mat.col(j) += lambda * arma::sign(B.col(j)); // add regulization
     }

     // Local Estimation
     eta = a * std::pow(1.0 / (t + 1), b); // stepsize
     Phi = B - eta * g_mat; // update the parameter


     B_old = B;
     B = Phi * C;

     // Calculate errors
     errors(t) = std::sqrt(std::pow(arma::norm(B - B_old, "fro"), 2))/arma::norm(B_old, "fro");
     if ((t > 0) && ((std::abs(errors(t) - errors(t - 1))) / errors(t - 1) < eps)) {
       break;
     }
   }

   if ((t + 1) <= (T - 1)) {
     errors.shed_rows(t + 1, T - 1);
   }

   return Rcpp::List::create(
     Rcpp::Named("B") = B,
     Rcpp::Named("history") = Rcpp::List::create(Rcpp::Named("errors") = errors));

 }

//' Implements decentralized subgradient descent (D-subGD) for SVM with BIC-regularized parameter selection.
//'
//' This function selects optimal lambda via BIC criterion from a regularization path.
//' Lambda values are logarithmically spaced from 1 to lambda_max and then multiplied lambda_factor.
//' For each lambda, trains a model using deSubGD_single_svm_real and computes BIC.
//' Final model uses lambda minimizing BIC = hinge loss + log(N)*sqrt(log(N))*shat.
//'
//' @param X Design matrix of size N × p (N total samples, p features).
//' @param y Response vector of length N (values should be +1 or -1).
//' @param adjacency_matrix Matrix defining communication links between nodes (m × m).
//' @param B_init Initial coefficient matrix of size p × m (one coefficient vector per node).
//' @param T Maximum number of iterations.
//' @param a Step size scaling parameter.
//' @param b Step size decay exponent.
//' @param nlambda Number of lambda values in regularization path (default=100).
//' @param lambda_factor Maximum lambda in the regularization path before multiplied scale.
//' @param lambda_max Multiplied scale of all lambda in the regularization path.
//' @param eps Convergence threshold for relative Frobenius norm change (default=1e-2).
//' @param quiet If true, suppresses console output (default=true).
//' @return A list containing (for optimal lambda):
//'   - `B`: Final estimated coefficient matrix (p × m)
//'   - `history`: List containing:
//'        - `errors`: Vector of relative RMSE at each iteration
//' @export
// [[Rcpp::export]]
Rcpp::List deSubGD_svm(arma::mat &X,
                        arma::vec &y,
                        arma::mat &adjacency_matrix,
                        arma::mat &B_init,
                        int T = 20,
                        double a = 3.5, // for stepsize
                        double b = 0.51, // for stepsize
                        int nlambda = 100L,
                        double lambda_factor = 1e-4,
                        double lambda_max = 1,
                        double eps = 1e-2,
                        bool quiet = true) {

   int m = adjacency_matrix.n_cols,
     N = X.n_rows,
     n = N / m;
   arma::mat B;
   arma::uvec idx(n);
   arma::vec y_temp(n);

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

   // Calculate BIC
   for (int ilambda = 0; ilambda < lambda_array.n_elem; ilambda++) {
     Rcpp::checkUserInterrupt();  // checking interruption
     lambda = lambda_array(ilambda);

     L = deSubGD_single_svm(X,
                            y,
                            adjacency_matrix,
                            B_init,
                            T,
                            a,
                            b,
                            lambda,
                            eps,
                            quiet);

     B = Rcpp::as<arma::mat>(L["B"]);


     // bic
     int shat = 0;
     for (int j = 0; j < m; j++)
     {
       idx = calN_j_cpp(n, j);
       y_temp = y(idx);
       arma::vec result = arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B.col(j));
       bic_array(ilambda) += arma::accu(pmax_arma( result, 0.0));
       shat = std::max(shat, int(arma::as_scalar(arma::accu(B.col(j).subvec(1, B.n_rows - 1) != 0))));
       }

     bic_array(ilambda) = bic_array(ilambda) + std::log(N) * std::sqrt(std::log(N)) * std::max(shat, 1);
     shat_array(ilambda) = shat;
   }

   arma::uword ilambda = arma::index_min(bic_array);
   lambda = lambda_array(ilambda);

   if (!quiet) {
     Rcpp::Rcout << "lambda = " << lambda << std::endl;
   }

   // Run algorithm with optimal lambda
   L = deSubGD_single_svm(X,
                          y,
                          adjacency_matrix,
                          B_init,
                          T,
                          a,
                          b,
                          lambda,
                          eps,
                          quiet);
   return L;
 }
