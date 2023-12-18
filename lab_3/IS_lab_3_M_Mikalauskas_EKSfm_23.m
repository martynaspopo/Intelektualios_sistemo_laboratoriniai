% IS-Lab3
% Aim: Learn to write training (parameter estimation) algorithm for 
% the Radial Basis Function Network based approximator's 2nd laye.

clc
close all

% Parametrai
learning_rate = 0.45;
epochs = 400000;

% Įėjimo duomenys
x = 0.1:1/22:1;
y_true = (1 + 0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2;

% Inicializuojame svorius ir poslinkį
w1 = randn;
w2 = randn;
w0 = randn;

c1 = 0.19; r1 = 0.15;
c2 = 0.87; r2 = 0.16;

% Tinklo mokymas
for epoch = 1:epochs
    for i = 1:length(x)
        % Įeinamasis signalas
        input = x(i);
        
        % Skaičiuojame spindulio tipo bazines funkcijas
        phi1 = exp(-(input - c1)^2 / (2*r1^2));
        phi2 = exp(-(input - c2)^2 / (2*r2^2));
        
        % Aproksimatoriaus funkcija
        output = w1 * phi1 + w2 * phi2 + w0;
        
        % Klaida
        error = y_true(i) - output;
        
        % Atnaujiname svorius
        w1 = w1 + learning_rate * error * phi1;
        w2 = w2 + learning_rate * error * phi2;
        w0 = w0 + learning_rate * error;
    end
end

% Testuojame tinklą su naujais duomenimis
x_test = 0.1:0.01:1;
y_pred = zeros(size(x_test));
y_trained = zeros(size(x));

for i = 1:length(x_test)
    input_test = x_test(i);
    phi1_test = exp(-(input_test - c1)^2 / (2*r1^2));
    phi2_test = exp(-(input_test - c2)^2 / (2*r2^2));
    y_pred(i) = w1 * phi1_test + w2 * phi2_test + w0;
end

for i = 1:length(x)
    input_test = x(i);
    phi1_test = exp(-(input_test - c1) ^ 2 / (2 * r1 ^2));
    phi2_test = exp(-(input_test - c2) ^ 2 / (2* r2 ^ 2));
    y_trained(i) = w1 * phi1_test + w2 * phi2_test + w0;
end


% Grafiko braižymas
figure(1);
plot(x, y_true, 'b', 'LineWidth', 2);
hold on;
plot(x_test, y_pred, 'r--', 'LineWidth', 2);
legend('Tikrasis atsakas', 'Prognozuotas atsakas');
xlabel('x');
ylabel('y');
title('Spindulio tipo bazinių funkcijų tinklo aproksimacija');
grid on;
plot(x, y_trained);


% % Nubrėžiame grafikus
% figure(2);
% plot(x, gauss1, 'b', 'LineWidth', 2, 'DisplayName', 'Gauss1');
% hold on;
% plot(x, gauss2, 'r', 'LineWidth', 2, 'DisplayName', 'Gauss2');
% xlabel('x');
% ylabel('Gauso funkcijos reikšmė');
% title('Gauso funkcijos su skirtingomis centrų ir spindulių reikšmėmis');
% legend;
% grid on;
