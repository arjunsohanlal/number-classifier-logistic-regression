%% Initialization
clear ; close all; clc

% Setting up parameters
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (mapped "0" to label 10 for simplicity)

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Training data stored in arrays X, y through MATLAB
% X is a 5000x400 matrix containing 5000 examples of handwritten data
% stored as 400-element vectors unwrapped from 20x20 pixel grayscale images
% y contains labels ranging from 1 to 10
load('trainingData.mat');
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

% One-vs-all training
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Predictions for accuracy test
pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);