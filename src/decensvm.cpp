#include <RcppArmadillo.h>
#include <cmath>
#include "decentralizedsvm.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Computes the cumulative distribution function (CDF) of the standard normal distribution.
//'
//' This function uses the complementary error function (`std::erfc`) to compute Φ(x).
//'
//' @param x Input value.
//' @return The value of the CDF at `x`.
// [[Rcpp::export]]
double Phi(double x) {
  return 0.5 * std::erfc(-x * std::sqrt(0.5));
}

//' Calculates the first derivative of the Gaussian loss function L_h^G(v).
//'
//' @param v Evaluation point.
//' @param h Bandwidth parameter.
//' @return The first derivative of the Gaussian loss function at `v`.
// [[Rcpp::export]]
double calL_h_G_prime(double v, double h) {
  return -Phi((1 - v) / h);
}

//' Calculates the second derivative of the Gaussian loss function L_h^G(v).
//'
//' @param v Evaluation point.
//' @param h Bandwidth parameter.
//' @return The second derivative of the Gaussian loss function at `v`.
// [[Rcpp::export]]
double calL_h_G_double_prime(double v, double h) {
  return (1 / (h * std::sqrt(2 * M_PI))) * std::exp(-std::pow(1 - v, 2) / (2 * std::pow(h, 2)));
}




//' Computes the gradient of the hinge loss for a specified kernel type.
//'
//' Supported types:
//' - 1: Uniform
//' - 2: Laplacian
//' - 3: Logistic
//' - 4: Gaussian
//' - 5: Epanechnikov
//'
//' @param Y Response vector.
//' @param X Design matrix.
//' @param beta Coefficient vector.
//' @param h Kernel bandwidth.
//' @param type Type of kernel (1–5).
//' @return Gradient of the hinge loss as an Armadillo vector.
//' @export
// [[Rcpp::export]]
arma::vec grad_hinge_loss(arma::vec& Y, arma::mat& X, arma::vec& beta, double h, int type) {
   if (type < 1 || type > 5) {
     Rcpp::stop("Invalid type parameter. Must be between 1 and 5.");
   }
   // Calculate gradient of Hinge Loss
   arma::vec v = Y % (X * beta);  // X * beta
   arma::vec grad_l(Y.n_elem);
   if (type == 1) {  // Uniform
     grad_l = arma::zeros<arma::vec>(Y.n_elem);
     for (arma::uword i = 0; i < Y.n_elem; ++i) {
       if (v(i) <= 1.0 - h) {
         grad_l(i) = -1.0;
       } else if (v(i) > 1.0 - h && v(i) <= 1.0 + h) {
         grad_l(i) = -(1.0 + h - v(i)) / (2.0 * h);
       } else {
         grad_l(i) = 0;
       }
     }
   } else if (type == 2) {  // Laplacian
     grad_l = arma::zeros<arma::vec>(Y.n_elem);
     for (arma::uword i = 0; i < Y.n_elem; ++i) {
       if (v(i) < 1) {
         grad_l(i) = -1.0 + 0.5 * exp((v(i) - 1) / h);
       } else {
         grad_l(i) = -0.5 * exp((1 - v(i)) / h);
       }
     }
   } else if (type == 3) {  // Logistic
     grad_l = arma::zeros<arma::vec>(Y.n_elem);
     for (arma::uword i = 0; i < Y.n_elem; ++i) {
       grad_l(i) = -1.0 + exp(v(i) / h) / (exp(1.0 / h) + exp(v(i) / h));
     }
   } else if (type == 4) {  // Gaussian
     grad_l = arma::zeros<arma::vec>(Y.n_elem);
     for (arma::uword i = 0; i < Y.n_elem; ++i) {
       grad_l(i) = calL_h_G_prime(v(i), h);
     }
   } else if (type == 5) {  // Epanechnikov
     grad_l = arma::zeros<arma::vec>(Y.n_elem);
     for (arma::uword i = 0; i < Y.n_elem; ++i) {
       if (v(i) <= 1.0 - h) {
         grad_l(i) = -1.0;
       } else if (v(i) > 1.0 - h && v(i) <= 1.0 + h) {
         grad_l(i) = -(1.0 + h - v(i)) * (1.0 + h - v(i)) * (2.0 * h - 1.0 + v(i)) / (4.0 * h * h * h);
       } else {
         grad_l(i) = 0;
       }
     }
   }
   return grad_l;
 }


//' Computes the L2-regularized hinge loss for a specific kernel type.
//'
//' Supported types:
//' - 1: Uniform
//' - 2: Laplacian
//' - 3: Logistic
//' - 4: Gaussian
//' - 5: Epanechnikov
//'
//' @param v Evaluation point.
//' @param h Kernel bandwidth.
//' @param type Type of kernel (1–5).
//' @return Value of the L2-regularized hinge loss at `v`.
//' @export
// [[Rcpp::export]]
double L2_HingeLoss(double v, double h, int type) {
   if (type == 1) {
     // Uniform Kernel
     if (v <= 1 - h || v > 1 + h) {
       return 0;
     } else {
       return 1 / (2 * h);
     }
   } else if (type == 2) {
     // Laplacian Kernel
     if (v < 1) {
       return std::exp((v - 1) / h) / (2 * h);
     } else {
       return std::exp((1 - v) / h) / (2 * h);
     }
   } else if (type == 3) {
     // Logistic Kernel
     double exp_v_h = std::exp(v / h);
     double exp_1_h = std::exp(1 / h);
     return (exp_v_h * exp_1_h) / (h * std::pow(exp_1_h + exp_v_h, 2));
   } else if (type == 4) {
     // Gaussian Kernel
     return std::exp(-std::pow(1 - v, 2) / (2 * std::pow(h, 2))) / (h * std::sqrt(2 * M_PI));
   } else if (type == 5) {
     // Epanechnikov Kernel
     if (v <= 1 - h || v > 1 + h) {
       return 0;
     } else {
       return (1 - v + h) * (3 * h - (1 - v)) / (2 * std::pow(h, 3));
     }
   } else {
     Rcpp::stop("Invalid kernel type. Please choose type = 1, 2, 3, 4, or 5.");
   }
 }



//' Calculates classification accuracy by comparing predicted labels with true labels.
//'
//' @param y True labels.
//' @param y_pred Predicted values.
//' @return Classification accuracy as a percentage.
//' @export
// [[Rcpp::export]]
double calculateAccuracy(arma::vec& y, arma::vec& y_pred) {

   // Step 1: Calculate the accuracy
   int n = y.n_elem;
   int correct = 0;

   for (int idx = 0; idx < n; idx++) {
     if (y[idx] == (y_pred[idx] > 0 ? 1 : -1)) {
       correct++;
     }
   }

   double accuracy = static_cast<double>(correct) / n;
   return accuracy;
 }


//' Finds indices of samples that belong to a specific partition.
//'
//' @param divisions Vector indicating which partition each sample belongs to.
//' @param division Partition number to extract.
//' @return Armadillo unsigned integer vector containing indices of samples in the specified partition.
//' @export
// [[Rcpp::export]]
arma::uvec getSampleIndices(const arma::vec& divisions, int division) {
   arma::uvec indices = arma::find(divisions == division);
   return indices;
}




//' Implements a decentralized SVM algorithm using ADMM optimization.
//'
//' Supports various kernel types and performs feature selection using soft thresholding.
//' Lambda values are logarithmically spaced from 1 to lambda_max and then multiplied lambda_factor.
//' Also calculates accuracy, BIC, and other performance metrics.
//'
//' @param X Design matrix of size N × p.
//' @param y Response vector of length N.
//' @param adjacency_matrix Matrix defining communication links between nodes.
//' @param B_init Initial coefficient matrix.
//' @param betaT True coefficient vector used for error calculation (ground truth).
//' @param h Kernel bandwidth.
//' @param type Kernel type (1–5).
//' @param C_N Regularization constant for BIC.
//' @param T_outer Number of iterations.
//' @param c0 Constant for step-size control (not used here but kept for consistency).
//' @param lambda_0 Regularization term of L2 penalty.
//' @param nlambda Number of regularization path points.
//' @param lambda_factor Maximum lambda in the regularization path before multiplied scale.
//' @param lambda_max Multiplied scale of all lambda in the regularization path.
//' @param tau_penalty_factor A penalty parameter in the augmented Lagrangian.
//' @param C_rho Scaling factor for rho_l update.
//' @param quiet Whether to suppress output messages.
//' @param eps Convergence tolerance (not currently used in stopping criterion).
//' @return A list containing:
//' - `B`: Final estimated coefficients (p × m matrix).
//' - `history`: List containing:
//'   - `errors_outer`: Frobenius norm errors over iterations.
//'   - `loss_error`: Hinge loss over iterations.
//'   - `shat_array`: Estimated sparsity (number of non-zero coefficients) over iterations.
//'   - `accuracy`: Classification accuracy over iterations.
//'  - `bic_mat`: BIC under different lambda.
//' @export
// [[Rcpp::export]]
Rcpp::List decentralizedsvm_cpp(arma::mat &X, arma::vec &y,
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
   arma::vec bic_array(nlambda - 1);
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


   if (!quiet)
   {
     Rcpp::Rcout << "accuracy =" << accuracy(0) << "\n";
   }


   arma::vec lambda_array(nlambda);
   arma::vec beta0 = arma::zeros<arma::vec>(p);
   lambda_array = arma::exp(arma::linspace(std::log(1), std::log(lambda_factor), nlambda));
   lambda_array.shed_row(0);
   lambda_array *= lambda_max;
   if(!quiet) {
     Rcpp::Rcout << "lambda_max = " << lambda_max << std::endl;
   }

   // ============================================================================
   // LAMBDA SELECTION VIA BIC
   // ============================================================================

   bic_array = arma::zeros(nlambda - 1, 1);
   arma::vec shat_array = arma::zeros(nlambda - 1, 1);
   arma::mat B_history  = arma::zeros<arma::mat>(p, m);
   double bic_opt;

   // Select optimal lambda
   double lambda_min;
   for (int ilambda = 0; ilambda < lambda_array.n_elem; ilambda++)
   {
     double lambda = lambda_array(ilambda);

     P_beta = arma::zeros(p, m);
     B_out = B_init;
     for (int v = 0; v < T_outer; v++)
     {
       B_out_old = B_out;
       Rcpp::checkUserInterrupt(); // checking interruption

       for (int j = 0; j < m; j++)
       {
         //int j = 1;
         idx = calN_j_cpp(n, j);
         y_temp = y(idx); //n*1
         X_temp = X.rows(idx); //n*p

         beta_temp = B_out_old.col(j); //p*1
         L_grad = grad_hinge_loss(y_temp, X_temp, beta_temp, h, type);

         arma::uvec neighbors_j = arma::find(adjacency_matrix.row(j));
         P_beta.col(j) = P_beta.col(j) + tau_penalty_beta *
           arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) - B_out_old.cols(neighbors_j), 1);

         arma::mat L_grad_broadcasted = arma::repmat(L_grad, 1, X_temp.n_cols);
         arma::mat y_temp_broadcasted = arma::repmat(y_temp, 1, X_temp.n_cols);
         tmp = omega(j) * ((rho(j)) * B_out_old.col(j) - 1.0 / n * arma::sum(L_grad_broadcasted % y_temp_broadcasted % X_temp, 0).t() - //lambda_0 *  B_out_old.col(j) -
           P_beta.col(j) + tau_penalty_beta * arma::sum(arma::repmat(B_out_old.col(j), 1, neighbors_j.n_elem) + B_out_old.cols(neighbors_j), 1));
         // Update B
         B_out(0,j) = tmp(0);
         arma::vec sub_tmp = tmp.subvec(1, tmp.n_elem - 1);
         B_out.col(j).subvec(1, tmp.n_elem - 1) = soft_thresholding_cpp(sub_tmp, lambda * omega(j));
       }

     }
     // Compute BIC for current lambda
     int shat = 0;
     for (int j = 0; j < m; j++)
     {
       idx = calN_j_cpp(n, j);
       y_temp = y(idx);
       arma::vec result = arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_out.col(j));
       bic_array(ilambda) += arma::accu(pmax_arma( result, 0.0));
       shat = shat + 1.0/m*double(arma::as_scalar(arma::accu(B_out.col(j).subvec(1, B_out.n_rows - 1) != 0)));
     }
     bic_array(ilambda) = bic_array(ilambda) + C_N * std::log(p) * (shat);
     shat_array(ilambda) = shat;

     if(ilambda == 0) {
       B_history = B_out;
       bic_opt = bic_array(ilambda);
     } else {
       if(bic_array(ilambda) < bic_opt) {
         B_history = B_out;
         bic_opt = bic_array(ilambda);
       }
     }
   }

   arma::uword optimal_idx = arma::index_min(bic_array);
   lambda_min = lambda_array(optimal_idx);
   int shat_min = shat_array(optimal_idx);


   if (!quiet)
   {
     Rcpp::Rcout<<"ilambda = "<< optimal_idx << "lambda = " << lambda_min << "bic_opt = " << bic_opt << "\n" << "shat = " << shat_min << std::endl;
   }


   B_out = B_init;
   P_beta = arma::zeros(p, m);
   for (int v = 0; v < T_outer; v++)
   {
     B_out_old = B_out;
     if (!quiet)
     {
       Rcpp::Rcout << "Outer iteration: v =" << v << ", h =" << h << "\n";
     }


     double bic_out = 0;
     for (int j = 0; j < m; j++)
     {
       idx = calN_j_cpp(n, j);
       y_temp = y(idx); //n*1
       X_temp = X.rows(idx); //n*p
       beta_temp = B_out_old.col(j); //p*1

       // Compute gradient
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
       B_out.col(j).subvec(1, tmp.n_elem - 1) = soft_thresholding_cpp(sub_tmp, lambda_min * omega(j));

       loss_error(j) = arma::accu(pmax_arma(arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_out.col(j)), 0.0));
       y_pred(idx) = X_temp * B_out.col(j);

       // //BIC
       arma::vec result = arma::ones<arma::vec>(y_temp.n_elem) - y(idx) % (X.rows(idx) * B_out.col(j));
       bic_out += arma::accu(pmax_arma( result, 0.0));
     }

     bic_out_array(v + 1) = bic_out/ m / n;
     accuracy(v + 1) = calculateAccuracy(y, y_pred);
     errors_inner(v + 1) = std::sqrt(std::pow(arma::norm(B_out - arma::repmat(betaT, 1, m), "fro"), 2) / m);
     if (!quiet)
     {
       Rcpp::Rcout << errors_inner(v) << "\t";
     }

   }


   return Rcpp::List::create(
     Rcpp::Named("B") = B_out,
     Rcpp::Named("B_history") = B_history,

     Rcpp::Named("history") = Rcpp::List::create(Rcpp::Named("errors_outer") =  errors_inner,
                 Rcpp::Named("loss_error") =  bic_out_array,
                 Rcpp::Named("accuracy") = accuracy,
                 Rcpp::Named("shat_mat") = shat_array,
                 Rcpp::Named("lambda_array") = lambda_array,
                 Rcpp::Named("bic_mat") = bic_array
     ));
 }
