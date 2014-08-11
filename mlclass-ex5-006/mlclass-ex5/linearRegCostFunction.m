function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h=X*theta;
error=h-y;
error_sqr=error.^2;
q=sum(error_sqr);
J1 = q/(2*m);
x=theta(2:end);

temp=(lambda*sum(x.^2))/2/m;
J=J1+temp;
sol1=(lambda.*theta.*[0;ones(length(theta)-1,1)]./m);
grad1 = (X'*(h-y))/m;
grad=grad1+sol1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
