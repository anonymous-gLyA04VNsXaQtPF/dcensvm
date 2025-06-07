#include <RcppArmadillo.h>
#include <cmath>
#include "decentralizedsvm.h"


// [[Rcpp::depends(RcppArmadillo)]]


//' Implements a decentralized penalized convoluted support vector machine (deCSVM) for real data analysis using ADMM optimization.
//'
//' This function performs decentralized training across multiple nodes defined by the adjacency matrix of real data.
//' It supports various kernel types for hinge loss computation and applies soft thresholding for feature selection.
//'
//' @param X Design matrix of size N × p (N total samples, p features).
//' @param y Response vector of length N.
//' @param divisions Real data divisions.
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
//' @param nlambda Number of regularization path points.
//' @param lambda_factor Maximum lambda in the regularization path before multiplied scale.
//' @param lambda_max Multiplied Scale of all lambda in the regularization path.
//' @param tau_penalty_factor Penalty factor for consensus term based on spectral norm.
//' @param C_rho Scaling factor for rho_l update.
//' @param quiet If true, suppresses console output.
//' @param eps Convergence tolerance (not currently used in stopping criterion).
//' @return A list containing:
//' - `B`: Final estimated coefficients (p × m matrix).
//' @export
// [[Rcpp::export]]
Rcpp::List decentralizedsvm_cpp_real(arma::mat &X, arma::vec &y,
                                      arma::vec &divisions,
                                      arma::mat &adjacency_matrix,
                                      arma::mat &B_init,
                                      arma::vec betaT,
                                      double h,
                                      int type,
                                      double C_N,
                                      int T_outer = 10,
                                      double c0 = 0.1,
                                      double tau_penalty_factor = 1 / 6,
                                      int nlambda = 100L,
                                      double lambda_factor = 1e-4,
                                      double lambda_max = 1,
                                      double lambda_0 = 0.5,
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
   default:
     Rcpp::stop("Invalid kernel type");
   }


   int m = adjacency_matrix.n_cols;
   int p = X.n_cols;  // dimension
   arma::vec bic_array(nlambda - 1);
   arma::vec lambda_array(nlambda);

   // sample index
   std::vector<arma::uvec> sample_indices(m);
   for (int j = 0; j < m; j++) {
     sample_indices[j] = getSampleIndices(divisions, j + 1);
   }


   arma::cube XtX(p, p, m, arma::fill::zeros);
   arma::vec rho(m, arma::fill::zeros);
   arma::vec omega(m, arma::fill::zeros);

   arma::vec beta_temp(p);
   arma::vec tau_penalty_beta(m);
   arma::vec tmp(p);

   for (int j = 0; j < m; j++) {
     arma::uvec idx = sample_indices[j];
     int n_j = idx.n_elem;  // the sample size pf

     if (n_j > 0) {
       arma::mat XtX_temp = arma::trans(X.rows(idx)) * X.rows(idx) / n_j;
       XtX.slice(j) = XtX_temp;

       arma::vec s_vec = arma::eig_sym(XtX_temp);
       rho(j) = C_rho * s_vec.max()*ch;
       tau_penalty_beta(j) = tau_penalty_factor;

       // calculate omega
       int neighbor_count = arma::accu(adjacency_matrix.row(j) != 0);  // neighbor numbers
       omega(j) = 1 / (2 * tau_penalty_factor * neighbor_count + rho(j) + lambda_0);
     } else {
       Rcpp::stop("Division has no samples.");
     }
   }

   double lambda_min;
   lambda_array = arma::exp(arma::linspace(std::log(1), std::log(lambda_factor), nlambda));
   lambda_array.shed_row(0);
   lambda_array *= lambda_max;
   if(!quiet) {
     Rcpp::Rcout << "lambda_max = " << lambda_max << std::endl;
   }
   bic_array = arma::zeros(nlambda - 1, 1);
   arma::vec shat_array = arma::zeros(nlambda - 1, 1);

   // arma::mat B_out_history  = arma::zeros<arma::mat>(p, m);
   double bic_opt;


   for (int ilambda = 0; ilambda < lambda_array.n_elem; ilambda++)
   {
     // // ===== Tune parameter ===== //
     double lambda = lambda_array(ilambda);

     arma::mat B_out(p, m, arma::fill::zeros),
     B_out_old(p, m, arma::fill::zeros),
     P_beta(p, m, arma::fill::zeros);
     B_out = B_init;

     // main
     for (int v = 0; v < T_outer; v++) {
       B_out_old = B_out;
       for (int j = 0; j < m; j++) {
         arma::uvec idx = sample_indices[j];
         int n_j = idx.n_elem;
         arma::vec y_temp = y(idx);
         arma::mat X_temp = X.rows(idx);

         beta_temp = B_out_old.col(j); //p*1
         arma::vec L_grad = grad_hinge_loss(y_temp, X_temp, beta_temp, h, type);
         arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
         P_beta.col(j) = P_beta.col(j) + tau_penalty_beta(j) *
           arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) - B_out_old.cols(neighbors_j), 1);
         arma::mat L_grad_broadcasted = arma::repmat(L_grad, 1, X_temp.n_cols); // 200x100
         arma::mat y_temp_broadcasted = arma::repmat(y_temp, 1, X_temp.n_cols); // 200x100
        tmp = omega(j) * ((rho(j)) * B_out_old.col(j) - 1.0 / n_j * arma::sum(L_grad_broadcasted % y_temp_broadcasted % X_temp, 0).t() -
           P_beta.col(j) + tau_penalty_beta(j) * arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) +
           B_out_old.cols(neighbors_j), 1));
         // Update B

         B_out(0,j) = tmp(0);
         arma::vec sub_tmp = tmp.subvec(1, tmp.n_elem - 1);
         B_out.col(j).subvec(1, tmp.n_elem - 1) = soft_thresholding_cpp(sub_tmp, lambda_max * omega(j));
       }
       // Check if B_out contains any NaN values
      if (B_out.has_nan()) {
        if (!quiet) {
          Rcpp::Rcout << "Warning: NaN detected in B_out at iteration " << v << ", lambda = " << lambda << ". Using B_init instead." << std::endl;
        }
        B_out = B_init;
        break;
      }

     }


     int shat = 0;
     for (int j = 0; j < m; j++)
     {
       arma::uvec idx = sample_indices[j];
       int n_j = idx.n_elem;
       arma::vec y_temp = y(idx);
       arma::mat X_temp = X.rows(idx);

       arma::vec result = arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_out.col(j));
       bic_array(ilambda) += arma::accu(pmax_arma( result, 0.0));
       shat = shat + 1.0/m *double(arma::as_scalar(arma::accu(B_out.col(j).subvec(1, B_out.n_rows - 1) != 0)));
     }
     bic_array(ilambda) = bic_array(ilambda) + C_N * std::log(p) * (shat);
     shat_array(ilambda) = shat;

     }



   arma::uword ilambda = arma::index_min(bic_array);
   lambda_min = lambda_array(ilambda);
   int shat_min = shat_array(ilambda);



   if (!quiet)
   {
     Rcpp::Rcout<<"ilambda = "<<ilambda << "lambda = " << lambda_min << "bic_opt = " << bic_opt << "\n" << "shat = " << shat_min << std::endl;
   }



   // Run the algorithm based on the optimal lambda
   arma::mat B_out(p, m, arma::fill::zeros),
   B_out_old(p, m, arma::fill::zeros),
   P_beta(p, m, arma::fill::zeros);
   B_out = B_init;

   // main
   for (int v = 0; v < T_outer; v++) {
     B_out_old = B_out;
     for (int j = 0; j < m; j++) {
       arma::uvec idx = sample_indices[j];
       int n_j = idx.n_elem;
       arma::vec y_temp = y(idx);
       arma::mat X_temp = X.rows(idx);

      beta_temp = B_out_old.col(j); //p*1
       arma::vec L_grad = grad_hinge_loss(y_temp, X_temp, beta_temp, h, type);
       arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
       P_beta.col(j) = P_beta.col(j) + tau_penalty_beta(j) *
         arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) - B_out_old.cols(neighbors_j), 1);
       arma::mat L_grad_broadcasted = arma::repmat(L_grad, 1, X_temp.n_cols); // 200x100
       arma::mat y_temp_broadcasted = arma::repmat(y_temp, 1, X_temp.n_cols); // 200x100
       tmp = omega(j) * ((rho(j)) * B_out_old.col(j) - 1.0 / n_j * arma::sum(L_grad_broadcasted % y_temp_broadcasted % X_temp, 0).t() -
         P_beta.col(j) + tau_penalty_beta(j) * arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) +
         B_out_old.cols(neighbors_j), 1));
       // Update B
       B_out(0,j) = tmp(0);
       arma::vec sub_tmp = tmp.subvec(1, tmp.n_elem - 1);
       B_out.col(j).subvec(1, tmp.n_elem - 1) = soft_thresholding_cpp(sub_tmp, lambda_min * omega(j));
     }
   }

   // Check if B_out contains any NaN values
   if (B_out.has_nan()) {
        B_out = B_init;
   }

   return Rcpp::List::create(
     Rcpp::Named("B") = B_out
   );
 }
