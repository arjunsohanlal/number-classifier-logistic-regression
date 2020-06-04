function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Evaluating hypothesis vector
h = sigmoid(X * theta);		% m x 1

% Evaluating cost function
J = (1/m) * (((-y)'*log(h)) - ((1-y)'*log(1-h))) + ((lambda/(2*m)) * (theta(2:end)' * theta(2:end)));

% Evaluating error list vector
errorList = h - y;			% m x 1

% Evaluating gradients vector
grad = (1/m) * X' * errorList;

% Adding regularization terms
reg_grad = [0; (lambda/m) * theta(2:end)];
grad = grad + reg_grad;

% Unwrapping gradients into single column vector
grad = grad(:);

end
