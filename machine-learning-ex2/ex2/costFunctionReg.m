function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hypothesis=sigmoid(X*theta);
temp1=(y'*log(hypothesis))+((1.-y')*log(1.-hypothesis));
len=size(theta);
temp2=lambda*sum(theta(2:len(1)).^2);
J=(temp1/-m)+temp2/(2*m);
for i = 1:m
	% hypothesis = mx1 column vector
	% y = mx1 column vector
	% X = mxn matrix
	grad = grad + ( hypothesis(i) - y(i) ) * X(i, :)';
end

RegularizationTerm = lambda/m * [0; theta(2:end)]; 
% where [0; theta(2:end)] is the same column vector theta beginning with a value of '0' at index
% 1 and then containing the old values from index 2:end of theta

% gradient = nx1 column vector
grad = (1/m) * grad + RegularizationTerm;


% =============================================================

end
