#ifndef DECENTRALIZEDSVM_H
#define DECENTRALIZEDSVM_H
#include <iostream>
#include <cmath>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]


arma::uvec calN_j_cpp(int n, int j);

arma::vec pmax_arma(arma::vec x, double bound);

arma::vec soft_thresholding_cpp(arma::vec x, double t);

arma::vec grad_hinge_loss(arma::vec& Y, arma::mat& X, arma::vec& beta, double h, int type);

arma::uvec getSampleIndices(const arma::vec& divisions, int division);

#endif
