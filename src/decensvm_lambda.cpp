#include <RcppArmadillo.h>
#include <cmath>
#include "decentralizedsvm.h"


// [[Rcpp::depends(RcppArmadillo)]]


double calculateAccuracy(arma::vec& y, arma::vec& y_pred);


//' Implements a decentralized penalized convoluted support vector machine (deCSVM) with fixed regularization parameter lambda using ADMM optimization.
//'
//' This function performs decentralized training across multiple nodes defined by the adjacency matrix.
//'
//' It supports various kernel types for hinge loss computation and applies soft thresholding for feature selection.
//' Unlike `decentralizedsvm_cpp`, this version uses a fixed regularization parameter `lambda`.
//' @param X Design matrix of size N × p (N total samples, p features).
//' @param y Response vector of length N.
//' @param adjacency_matrix Matrix defining communication links between nodes (m × m).
//' @param B_init Initial coefficient matrix of size p × m (one coefficient vector per node).
//' @param betaT True coefficient vector used for error calculation (ground truth).
//' @param h Kernel bandwidth.
//' @param type Type of kernel function to use (1: Uniform, 2: Laplacian, 3: Logistic, 4: Gaussian, 5: Epanechnikov).
//' @param C_N Regularization constant used in BIC calculation.
//' @param lambda Fixed L1 regularization parameter.
//' @param T_outer Maximum number of iterations.
//' @param c0 Constant for step-size control (not used here but kept for consistency).
//' @param lambda_0 Regularization term of L2 penalty.
//' @param tau_penalty_factor A penalty parameter in the augmented Lagrangian.
//' @param C_rho Scaling factor for rho_l update.
//' @param quiet If true, suppresses console output.
//' @param eps Convergence tolerance (not currently used in stopping criterion).
//' @return A list containing:
//' - `B`: Final estimated coefficients (p × m matrix).
//' - `history`: List containing:
//'   - `errors_outer`: Frobenius norm errors over iterations.
//'   - `loss_error`: Hinge loss over iterations.
//'   - `shat_array`: Estimated sparsity (number of non-zero coefficients) over iterations.
//'   - `accuracy`: Classification accuracy over iterations.
//' @export
// [[Rcpp::export]]
Rcpp::List decentralizedsvm_lambda(arma::mat &X, arma::vec &y,
                                    arma::mat &adjacency_matrix,
                                    arma::mat &B_init,
                                    arma::vec betaT,
                                    double h,
                                    int type,
                                    double C_N,
                                    double lambda,
                                    int T_outer = 10,
                                    double c0 = 0.1,
                                    double lambda_0 = 0,
                                    double tau_penalty_factor = 1 / 6,
                                    double C_rho = 1.01,
                                    bool quiet = true,
                                    double eps = 1e-2) //kernel type
 {
   //double kappa;
   double ch;
   //calculate ch
   switch (type) {
   case 1:  // Uniform
     ch = 1.0 / (2.0 * h);
     break;
   case 2:  // Laplacian
     ch = 1.0 / (2.0 * h);
     break;
   case 3:  // Logistic
     ch = 1.0 / (4.0 * h);
     break;
   case 4:  // Gaussian
     ch = 1 / (h * std::sqrt(2 * M_PI));
     break;
   case 5:  // Epanechnikov
     ch = 3.0 / (4.0 * h);
     break;
   }
   // Parameters
   int m = adjacency_matrix.n_cols, p = X.n_cols, N = X.n_rows;
   int n = N / m;
   arma::vec s_vec;
   double tau_penalty_beta = tau_penalty_factor;







   // Results
   arma::mat B_out(p, m, arma::fill::zeros),
   B_out_old(p, m, arma::fill::zeros),
   P_beta(p, m, arma::fill::zeros);
   arma::uvec idx(n);

   arma::vec errors_inner(T_outer + 1, arma::fill::zeros);
   arma::vec accuracy(T_outer + 1, arma::fill::zeros);
   arma::vec bic_out_array(T_outer + 1, arma::fill::zeros);
   arma::vec shat_array(T_outer + 1, arma::fill::zeros);


   // cache covariance
   arma::cube XtX(p, p, m, arma::fill::zeros); //XtX
   for (int j = 0; j < m; j++)
   {
     idx = calN_j_cpp(n, j);
     XtX.slice(j) = arma::trans(X.rows(idx)) * X.rows(idx);
   }


   // cache omega and rho
   arma::vec rho(m), omega(m);
   arma::mat XtX_temp(p, p);
   arma::vec y_temp(n);
   arma::vec beta_temp(p);
   arma::mat X_temp(p,p);


   for (int j = 0; j < m; j++)
   {
     idx = calN_j_cpp(n, j);
     XtX_temp = XtX.slice(j);
     s_vec = arma::eig_sym(XtX_temp /n);
     rho(j) = C_rho * s_vec.back()*ch;
     omega(j) = 1 / (2 * tau_penalty_beta * arma::accu(adjacency_matrix.row(j)!=0) + rho(j) + lambda_0);
   }




   // === Main routine === //
   arma::vec tmp(p);
   arma::vec L_grad(n);

   arma::vec loss_error = arma::zeros<arma::vec>(m);
   arma::vec loss_error0 = arma::zeros<arma::vec>(m);
   arma::vec y_pred = arma::zeros<arma::vec>(n*m);
   arma::vec y_pred0 = arma::zeros<arma::vec>(n*m);




   // initial error
   for (int j = 0; j < m; j++){
     idx = calN_j_cpp(n, j);
     if (arma::any(idx >= X.n_rows)) {
       Rcpp::stop("Index out of bounds in calN_j_cpp");
     }
     y_temp = y(idx); //n*1
     X_temp = X.rows(idx); //n*p
     loss_error0(j) = arma::accu(pmax_arma(arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_init.col(j)), 0.0));
     y_pred0(idx) = X_temp * B_init.col(j);
   }

   bic_out_array(0) = arma::accu(loss_error0) / m / n;
   errors_inner(0) = std::sqrt(std::pow(arma::norm(B_init - arma::repmat(betaT, 1, m), "fro"), 2) / m);
   accuracy(0) = calculateAccuracy(y, y_pred0);


   // if (!quiet)
   // {
   //   Rcpp::Rcout << "accuracy =" << accuracy(0) << "\n";
   // }
   //
   if (!quiet)
   {
     Rcpp::Rcout << "lambda = " << lambda << std::endl;
   }

   B_out = B_init;
   P_beta = arma::zeros(p, m);
   for (int v = 0; v < T_outer; v++)
   {
     B_out_old = B_out;
     // if (!quiet)
     // {
     //   Rcpp::Rcout << "Outer iteration: v =" << v << ", h =" << h << "\n";
     // }


     double bic_out = 0;
     double shat = 0;
     for (int j = 0; j < m; j++)
     {
       idx = calN_j_cpp(n, j);
       y_temp = y(idx); //n*1
       X_temp = X.rows(idx); //n*p
       beta_temp = B_out_old.col(j); //p*1
       L_grad = grad_hinge_loss(y_temp, X_temp, beta_temp, h, type);
       arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
       P_beta.col(j) = P_beta.col(j) + tau_penalty_beta *
         arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) - B_out_old.cols(neighbors_j), 1);
       arma::mat L_grad_broadcasted = arma::repmat(L_grad, 1, X_temp.n_cols); // 200x100
       arma::mat y_temp_broadcasted = arma::repmat(y_temp, 1, X_temp.n_cols); // 200x100
      tmp = omega(j) * (rho(j) * B_out_old.col(j) - 1.0 / n * arma::sum(L_grad_broadcasted % y_temp_broadcasted % X_temp, 0).t() -
        P_beta.col(j) + tau_penalty_beta * arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) + B_out_old.cols(neighbors_j), 1));
       // Update B
       B_out(0,j) = tmp(0);
       arma::vec sub_tmp = tmp.subvec(1, tmp.n_elem - 1);
       B_out.col(j).subvec(1, tmp.n_elem - 1) = soft_thresholding_cpp(sub_tmp, lambda * omega(j));
       loss_error(j) = arma::accu(pmax_arma(arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_out.col(j)), 0.0));
       y_pred(idx) = X_temp * B_out.col(j);

       // BIC
       arma::vec result = arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_out.col(j));
       bic_out += arma::accu(pmax_arma( result, 0.0));
       shat = shat + 1.0/m*double(arma::as_scalar(arma::accu(B_out.col(j).subvec(1, B_out.n_rows - 1) != 0)));
     }
     bic_out_array(v + 1) = bic_out/ m / n + C_N * std::log(p) * (shat);
     shat_array(v + 1) = shat;

     accuracy(v + 1) = calculateAccuracy(y, y_pred);
     errors_inner(v + 1) = std::sqrt(std::pow(arma::norm(B_out - arma::repmat(betaT, 1, m), "fro"), 2) / m);
     if (!quiet)
     {
       Rcpp::Rcout << errors_inner(v) << "\t";
     }

   }



   return Rcpp::List::create(
     Rcpp::Named("B") = B_out,
     Rcpp::Named("history") = Rcpp::List::create(Rcpp::Named("errors_outer") =  errors_inner,
                 Rcpp::Named("loss_error") =  bic_out_array,
                 Rcpp::Named("shat_array") = shat_array,
                 Rcpp::Named("accuracy") = accuracy
     ));
 }
