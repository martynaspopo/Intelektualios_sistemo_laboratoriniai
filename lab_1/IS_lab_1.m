clear variables
clc

% Classification using perceptron

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1);          %color
metric_A1=apvalumas_roundness(A1);      %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2);          %color
metric_A2=apvalumas_roundness(A2);      %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3);          %color
metric_A3=apvalumas_roundness(A3);      %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4);          %color
metric_A4=apvalumas_roundness(A4);      %roundness
% 5th apple image(A5)                   
metric_A5=apvalumas_roundness(A5);      %roundness
hsv_value_A5=spalva_color(A5);          %color
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6);          %color
metric_A6=apvalumas_roundness(A6);      %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7);          %color
metric_A7=apvalumas_roundness(A7);      %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8);          %color
metric_A8=apvalumas_roundness(A8);      %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9);          %color
metric_A9=apvalumas_roundness(A9);      %roundness

% For Pears
% 1st pear image(P1)
hsv_value_P1=spalva_color(P1);          %color
metric_P1=apvalumas_roundness(P1);      %roundness
% 2nd pear image(P2)
hsv_value_P2=spalva_color(P2);          %color
metric_P2=apvalumas_roundness(P2);      %roundness
% 3rd pear image(P3)
hsv_value_P3=spalva_color(P3);          %color
metric_P3=apvalumas_roundness(P3);      %roundness
% 2nd pear image(P4)
hsv_value_P4=spalva_color(P4);          %color
metric_P4=apvalumas_roundness(P4);      %roundness

% Selecting features(color, roundness, 3 apples and 2 pears)
% A1,A2,A3,P1,P2
% Building train matrix 2x5
x1=[hsv_value_A1 hsv_value_P1 hsv_value_A2 hsv_value_A3 hsv_value_P2 ];
x2=[metric_A1 metric_P1 metric_A2 metric_A3 metric_P2];

%x1=[hsv_value_A1 hsv_value_P1 hsv_value_A2 hsv_value_P2];
%x2=[metric_A1 metric_P1 metric_A2 metric_P2];

% Building test data matrix
x1_test=[hsv_value_A4 hsv_value_P3 hsv_value_A5 hsv_value_A6 hsv_value_P4];
x2_test=[metric_A4 metric_P3 metric_A5 metric_A6 metric_P4];

% Estimated features are stored in matrix P:
P=[x1; x2];

% Test data vector 
P_test = [x1_test; x2_test];

% Desired output vector of labled output data
T=[1; -1; 1; 1; -1; 1]; 

% Desired labled test output data vector
T_test = [1; -1; 1; 1; -1];

%----------------------------------------------------------
% Train single perceptron with two inputs and one output
%----------------------------------------------------------

% Generate random initial values of w1, w2 and b in range 0...1
w1 = randn(1);
w2 = randn(1);
b = randn(1);
% Define cycle varaibles
e = zeros(1, 5);
e_total = 0;

%Error calculation no traininig

for i = 1:size(P, 2)
    % Calculate weighted sum with randomly generated parameters
    v = w1 * P(1, i) + w2 * P(2, i) + b;
    % Calculate current output of the perceptron 
    if v > 0  
	    y = 1;
    else
	    y = -1;
    end
    % Calculate the error
    e(i) = T(i, 1) - y;
    
    % Calculate the total error for these 5 inputs 
    e_total = e_total + abs(e(i));

end

% Perceptron training part

l_rate = 0.1;         % Set the learning rate 0 < eta < 1
max_epoch_num = 500;  % Set the maximum number of training epochs
e_total = 1;          % initialize the total error
epoch = 0;            % Initialize the epoch cpunter

% Calculating new wheights
while e_total ~= 0 && (epoch < max_epoch_num)
	
    e_total = 0;

    for i = 1:size(P, 2)
        % Calculate the weighted sum
        v = w1 * P(1, i) + w2 * P(2, i) + b;        
        % Calculate the current output of the perceptron
        if v > 0
            y = 1;
        else
            y = -1;
        end
        
        % Calculate the error for the current example
        e = T(i, 1) - y;       
        % Update weights and bias
        w1 = w1 + l_rate * e * P(1, i);
        w2 = w2 + l_rate * e * P(2, i);
        b = b + l_rate * e;
        
        % Accumulate the error for this epoch
        e_total = e_total + abs(e);
    end
    
    % Increment the epoch counter
    epoch = epoch + 1;
       
end
fprintf('Training: ');
fprintf('\n');
fprintf('w1: %f  w2 %f b %f\nEpoch: %d Total Error = %f\n', w1, w2, b, epoch, e_total);
fprintf('\n');
v_test = zeros(1, 5);
y_test = zeros(1, 5);

% Evaluate new wheights with test data set
    for i = 1:size(P_test, 2)
       
        v_test(i) = w1 * P_test(1, i) + w2 * P_test(2, i) + b;        
       
        if v_test(i) > 0
            y_test(i) = 1;
        else
            y_test(i) = -1;
        end
    
        e(i) = T_test(i, 1) - y_test(i);
      
    end
 
% Calculate outputs and errors for all examples
% using current values of the parameter set {w1, w2, b}
fprintf('Testing new data: ');
fprintf('\n');

fprintf('v: ');
fprintf('%f ', v_test);
fprintf('\n');

fprintf('y: ');
fprintf('%f ', y_test);
fprintf('\n');

fprintf('e: ');
fprintf('%f ', e);
fprintf('\n');

% ------------------------------------------------------------   
% Naive Bayes clasifier implementation
% ------------------------------------------------------------
fprintf('\n');
fprintf('Test naive bayes: ');
fprintf('\n');
% Define labled learning data vector for NB clasifier
NB_train_set = [1; -1; 1; 1; -1];

% Define another labled data vector to test clasifier
NB_test_set = [1; -1; 1; 1; -1];

% Train the Naive Bayes classifier 
% Create nbClassifier object which encapsulates Naive Bayes model
nbClassifier = fitcnb(transpose(P), NB_train_set);

% Predict labels for the testing data
predicted_labels = predict(nbClassifier, transpose(P));
disp(transpose(predicted_labels));


    
