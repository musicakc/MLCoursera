function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
h=sigmoid(X*theta);

term1=-y'*log(h);
term2=(1-y')*log(1-h);
x=theta(2:end);

temp=(lambda*sum(x.^2))/2/m;
sol1=(lambda.*theta.*[0;ones(length(theta)-1,1)]./m);

% You need to return the following variables correctly 
J = ((term1-term2)/m)+temp;
grad1 = (X'*(h-y))/m;
grad=grad1+sol1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
