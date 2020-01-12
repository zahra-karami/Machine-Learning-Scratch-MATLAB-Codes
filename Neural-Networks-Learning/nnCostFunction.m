function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(size(X,1), 1) X]'; % X + bias unit for each training set

z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1,size(a2,2)); a2];

z3 = Theta2 * a2;
h_theta = sigmoid(z3);

K = num_labels;
y_k = eye(K);
cost = zeros(K,1);

for i=1:m
    value = -y_k(:,y(i)) .* log(h_theta(:,i)) -(1 - y_k(:,y(i))) .* log(1 - h_theta(:,i)) ;
    cost = cost + value ;
end

J = sum(cost) / m;
regularizationTerm = sum(sum(Theta1(:,2:end) .^2)) + sum(sum(Theta2(:,2:end).^2)); 
J = J + regularizationTerm * lambda / (2 * m);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

Delta1_2 = zeros(size(Theta2));
Delta1_1 = zeros(size(Theta1));

for t=1:m

	a1 = [1 ; X(t,:)'];
 
	z2 = Theta1 *a1 ;
	a2 = sigmoid(z2);
	a2 = [1;a2];
 
	z3 = Theta2 *a2;
	a3 = sigmoid(z3);

	delta_3 = a3 - y_k(:,y(t));
	delta_2 = (Theta2' * delta_3 ) .* [0 ; sigmoidGradient(z2)];
	delta_2 = delta_2(2:end,1);
	
	Delta1_2 = Delta1_2 +   delta_3 * a2';   
	Delta1_1 = Delta1_1 +   delta_2 * a1';
end

Theta1_grad = Delta1_1 /m + ( lambda / m ) * [zeros(size(Theta1,1),1) , Theta1(:,2:end)];
Theta2_grad = Delta1_2 /m + ( lambda / m ) * [zeros(size(Theta2,1),1) , Theta2(:,2:end)];  


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
