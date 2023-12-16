close all
clear variables
clc
%% simbolių pavyzdžių nuskaitymas ir požymių skaičiavimas
pavadinimas = 'train_10_final.png';

pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, 9);
%% Skaičių atpažintuvo kūrimas
% požymiai iš celių masyvo perkeliami į matricą
P = cell2mat(pozymiai_tinklo_mokymui);
% sukuriama teisingų atsakymų matrica: 10 simboliu, 9 eilutës mokymui

T = [eye(10), eye(10), eye(10), eye(10), eye(10) , eye(10), eye(10), eye(10), eye(10)];
% sukuriamas RBF tinklas klasifikuoti duotiems sąryšiams P ir T, naudojant 10 neuronų
tinklas = newrb(P, T, 0, 1, 10);

%% Tinklo patikra
% skaièiuojamas tinklo išëjimas neþinomiems poþymiams

P2 = P(:,1:10);
Y2 = sim(tinklas, P2);
% ieškoma, kuriame išėjime gauta didžiausia reikšmë

[a2, b2] = max(Y2);
%% Rezultato atvaizdavimas
% skaičiuojamas simboliu sk eilutėje

simbol_sk = size(P2,2);
% rezultatą saugosime kintamajame 'simboliai'
simboliai = [];
for k = 1:simbol_sk
    switch b2(k)

        case 1           
            simboliai = [simboliai, '0'];
        case 2
            simboliai = [simboliai, '1'];
        case 3
            simboliai = [simboliai, '2'];
        case 4
            simboliai = [simboliai, '3'];
        case 5
            simboliai = [simboliai, '4'];
        case 6
            simboliai = [simboliai, '5'];
        case 7
            simboliai = [simboliai, '6'];
        case 8
            simboliai = [simboliai, '7'];
        case 9
            simboliai = [simboliai, '8'];
        case 10
            simboliai = [simboliai, '9'];
        
    end
end

% Rezultatai pateikiami komandiniame lange
disp(simboliai)


%% Telefono numerio atpažinimas

pavadinimas = 'phone_num.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

% požymiai iš celių masyvo perkeliami į matricą
% features from cell-variable are stored to matrix-variable
P2 = cell2mat(pozymiai_patikrai);
% skaičiuojamas tinklo išėjimas nežinomiems požymiams
Y2 = sim(tinklas, P2);
% ieškoma, kuriame išėjime gauta didžiausia reikmė
[a2, b2] = max(Y2);

%% Rezultato atvaizdavimas
% apskaičiuosime simbolių skaičių - požymių P2 stulpelių skaičių
simbol_sk = size(P2,2);
% Nuskaitytas numeris saugojamas kintamajame numeris
simboliai = [];
for k = 1:simbol_sk
    switch b2(k)

        case 1           
            simboliai = [simboliai, '0'];
        case 2
            simboliai = [simboliai, '1'];
        case 3
            simboliai = [simboliai, '2'];
        case 4
            simboliai = [simboliai, '3'];
        case 5
            simboliai = [simboliai, '4'];
        case 6
            simboliai = [simboliai, '5'];
        case 7
            simboliai = [simboliai, '6'];
        case 8
            simboliai = [simboliai, '7'];
        case 9
            simboliai = [simboliai, '8'];
        case 10
            simboliai = [simboliai, '9'];

    end
end

% Telefono numeris pateikiamas 
disp(simboliai)

%% Papildoma dalis - apmokymui naudojama matlab feedforwardnet funkcija

%% Įpatybių nuskaitymas
pavadinimas = 'train_10_final.png';
pozymiai_tinklo_mokymui = pozymiai_raidems_atpazinti(pavadinimas, 9);

% Extract features from cell array and concatenate into a matrix
P = cell2mat(pozymiai_tinklo_mokymui);

% Neuronu skaičiu paslėptame sluoksnyje
hiddenlayer_num = 35; 

% feedforward tinklo kūrimas
net = feedforwardnet(hiddenlayer_num);

% Mokymo parametrai
net.trainParam.epochs = 100000; % Jei reikia, kaitalioti
net.trainParam.lr = 0.05;       % Jie reikia, kaitalioti

% Apmokymas
net = train(net, P, T);

% Patikra
pavadinimas = 'phone_num.png';
pozymiai_patikrai = pozymiai_raidems_atpazinti(pavadinimas, 1);

% Skaičių atpažintuvo kūrimas
% požymiai iš celių masyvo perkeliami į matricą
P = cell2mat(pozymiai_patikrai);

Y = sim(net, P);
[a, b] = max(Y);
simbol_sk = size(P,2);

% Nuskaitytas numeris saugojamas kintamajame simboliai
simboliai = [];
for k = 1:simbol_sk
    switch b(k)

        case 1           
            simboliai = [simboliai, '0'];
        case 2
            simboliai = [simboliai, '1'];
        case 3
            simboliai = [simboliai, '2'];
        case 4
            simboliai = [simboliai, '3'];
        case 5
            simboliai = [simboliai, '4'];
        case 6
            simboliai = [simboliai, '5'];
        case 7
            simboliai = [simboliai, '6'];
        case 8
            simboliai = [simboliai, '7'];
        case 9
            simboliai = [simboliai, '8'];
        case 10
            simboliai = [simboliai, '9'];
    end
end

% Telefono numeris pateikiamas 
disp(simboliai)


