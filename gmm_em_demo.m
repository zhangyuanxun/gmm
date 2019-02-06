clear all;
% load 2D dataset as X
% load('data.mat');

K = 5;                 % number of cluster
N = 1000;
prior_var = 2500;

X = genMixtureGaussian(N, K, [0, 0], prior_var);           
em(X, K);             % run em algorithm