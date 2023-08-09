function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ===================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

J_history(iter) = computeCost(X,y,theta);
%disp(J_history(iter));
pred = (X*theta);
error = pred - y;
delta = (error'*X)';
theta = theta - alpha*(1/m)*delta;
%temp1 = theta(2) - alpha*(1/m)*delta(2);
%iter = iter + 1 ;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
%theta(1) = temp0;
%theta(2) = temp1;
end
