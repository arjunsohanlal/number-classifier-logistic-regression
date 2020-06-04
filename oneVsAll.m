function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Converting single number labels to binary label vectors
% Initializing said vectors to zeros
tr = zeros(num_labels,m);

% Running training for each classifier from 1 - 10
for c = 1:num_labels
	tr(c,:) = (y == c)';
	initial_theta = zeros(n + 1, 1);
	options = optimset('GradObj', 'on', 'MaxIter', 50);
	[theta, fX, iter] = fmincg(@(t)(lrCostFunction(t, X, tr(c,:)', lambda)), initial_theta, options);
	all_theta(c,:) = theta';
	fprintf('Completed iteration %d.\n',c);
end

end
