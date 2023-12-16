%% Task description
% Create a program to calculate the coefficients of the multilayer perceptron. 
% The multilayer perceptron must perform the function of an approximator. 
% Structure of the multilayer perceptron: 
% one input (input 20 input vectors (20 examples) X, with values in the 
% range 0 to 1, eg x = 0.1: 1/22: 1;).
% one output (for example, the output is expecting the desired response 
% that can be calculated using the formula: y = (1 + 0.6 * sin (2 * pi * x / 0.7)) + 0.3 * sin (2 * pi * x)) / 2; 
% - the neural network being created should "model / simulate the behavior 
% of this formula" using a completely different mathematical expression than this);
% One hidden layer with hyperbolic tangent or sigmoidal activation functions in neurons 
% (number of neurons: 4-8);
% linear activation function in the output neuron;
% training algorithm - Backpropagation.
% Aditional: Solve the surface approximation task (two inputs and single output).
%%

clc 
close all
clear variables
% Duomenų įvesties į tinklą vektorius
x = 0.1:1/22:1;
% Vektorius apmokytoms funkcijos reikšmėms gauti
y_trained = zeros(1, 20);
% Norima aproksmituoti funkcija/sugeneruojami apmokymo taškai
y = (1 + 0.6 * sin (2 * pi * x / 0.7)) + 0.3 * sin (2 * pi * x) / 2;
% d - koeficientas pakeisti pradines svorius vrrtes vienu metu.
d = 1;

figure(1)
plot(x,y, 'kx');
hold on;

%% 2. Tinklo struktūros pasirinkimas
% vienas įėjimas; 5 neuronai paslėptajame sluoksnyje, 1 išėjimas
% paslėptajame sluoksnyje sigmoidė, išėjimo sluoksnyje – tiesinė aktyvavimo
% funkcija
% Mokymo algoritmas - Backpropogation (atgalinio sklidimo)
%% 3. Pradinių parametrų reikšmių pasirinkimas
% I sluoksnis
w11_1 = randn(1)/d; b1_1 = randn(1)/d;
w12_1 = randn(1)/d; b2_1 = randn(1)/d;
w13_1 = randn(1)/d; b3_1 = randn(1)/d;
w14_1 = randn(1)/d; b4_1 = randn(1)/d;
w15_1 = randn(1)/d; b5_1 = randn(1)/d;
% II sluoksnis
w11_2 = randn(1)/d; b1_2 = rand(1)/d;
w21_2 = randn(1)/d; 
w31_2 = randn(1)/d; 
w41_2 = randn(1)/d; 
w51_2 = randn(1)/d; 
%%
% n - apmokymo sparta
n = 0.4;
% Maksimalus epochų skaičius
max_epoch_num = 40000;
% Skaičius sekti treniravimo epochų skaičiui
epoch = 0;  

% Apmokymas vykdomas iki kol pasiekiamas maksimalus epochų skaičius
while epoch < max_epoch_num
    
    for i = 1:length(x)
        % Neuornų paslėptame sluolksnyje atsako skaičiavimas
        v1_1 = x(i)*w11_1 + b1_1; y1_1 = 1/(1+exp(-v1_1));
        v2_1 = x(i)*w12_1 + b2_1; y2_1 = 1/(1+exp(-v2_1));
        v3_1 = x(i)*w13_1 + b3_1; y3_1 = 1/(1+exp(-v3_1));
        v4_1 = x(i)*w14_1 + b4_1; y4_1 = 1/(1+exp(-v4_1));
        v5_1 = x(i)*w15_1 + b5_1; y5_1 = 1/(1+exp(-v5_1));
        % Išėjimo sluoksnio atsakas
        v1_2 = y1_1*w11_2 + y2_1*w21_2 + y3_1*w31_2 + y4_1*w41_2 + y5_1*w51_2 + b1_2; y1 = v1_2;
    
        % Klaidos skaičiavimas
        e1 = y(i) - y1;
        y_trained(i) = y1;
        
        % Išėjimo sluoksnio gradientas gaunamas atlikus tikslo funkcijos
        % išvestinę
        delta_out1 = 1 * 1/2 * 2 * e1;
        % Kiekvieno iš neuronų įtaka išėjimo klaidai
        delta_hidden1 = y1_1 * (1 - y1_1)*(w11_2 * delta_out1);
        delta_hidden2 = y2_1 * (1 - y2_1)*(w21_2 * delta_out1);
        delta_hidden3 = y3_1 * (1 - y3_1)*(w31_2 * delta_out1);
        delta_hidden4 = y4_1 * (1 - y4_1)*(w41_2 * delta_out1);
        delta_hidden5 = y5_1 * (1 - y5_1)*(w51_2 * delta_out1);
        % Tinklo koeficientų (ryšių svorių) atnaujinimas
        w11_2 = w11_2 + n * delta_out1 * y1_1; 
        w12_2 = w21_2 + n * delta_out1 * y2_1;
        w13_2 = w31_2 + n * delta_out1 * y3_1; 
        w14_2 = w41_2 + n * delta_out1 * y4_1;
        w15_2 = w51_2 + n * delta_out1 * y5_1;
        b1_2 = b1_2 + n * delta_out1 * 1; 
    
        % Atnaujinimas pirmame sluoksnyje
        w11_1 = w11_1 + n * delta_hidden1 * x(i);
        b1_1 = b1_1 + n * delta_hidden1;
        w12_1 = w12_1 + n * delta_hidden2 * x(i); 
        b2_1 = b2_1 + n * delta_hidden2;
        w13_1 = w13_1 + n * delta_hidden3 * x(i);
        b3_1 = b3_1 + n * delta_hidden3;
        w14_1 = w14_1 + n * delta_hidden4 * x(i); 
        b4_1 = b4_1 + n * delta_hidden4;
        w15_1 = w15_1 + n * delta_hidden5 * x(i); 
        b5_1 = b5_1 + n * delta_hidden5;
    
    end  

    epoch = epoch + 1;
   
end

plot(x, y_trained, 'bo');
title('Apmokymo rezultatai');
legend('Tikrieji taškai','Modelio aproksimacija','Location','NorthEast')
xlabel('x');
ylabel('y');
hold off;

% Sukuriamas naujas įėjimo duomenų vektorius siekiant ištestuoti modelį.
x_test = 0.1:1/250:1;
% Analogiškai sukuriamas išėjimo vektorius.
Y_tested = zeros(1, length(x_test));
Y_real = zeros(1, length(x_test));
e1  = 0;

for i = 1:length(x_test)

    % Neuornų paslėptame sluolksnyje atsako skaičiavimas
    v1_1 = x_test(i) * w11_1 + b1_1; y1_1 = 1/(1+exp(-v1_1));
    v2_1 = x_test(i) * w12_1 + b2_1; y2_1 = 1/(1+exp(-v2_1));
    v3_1 = x_test(i) * w13_1 + b3_1; y3_1 = 1/(1+exp(-v3_1));
    v4_1 = x_test(i) * w14_1 + b4_1; y4_1 = 1/(1+exp(-v4_1));
    v5_1 = x_test(i) * w15_1 + b5_1; y5_1 = 1/(1+exp(-v5_1));
    % Išėjimo sluoksnio atsakas
    v1_2 = y1_1 * w11_2 + y2_1 * w21_2 + y3_1 * w31_2 + y4_1 * w41_2 + y5_1 * w51_2 + b1_2; y1 = v1_2;
    Y_tested(i) = y1;
     
end

figure(2)
plot(x,y, 'kx');
hold on
plot(x_test, Y_tested)
title('Apmokyto modelio testavimas');
xlabel('x');
ylabel('y');
legend('Tikrieji taškai','Modelio aproksimacija','Location','NorthEast')
hold off
