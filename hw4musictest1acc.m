% Test Model Classification LD
clear; close all; clc;

mp3Fs = 44100; % Sample Rate, Hz

mdir = 'music';
D = dir(mdir);
D = D(3:end);

trials = zeros([10 1]);

for i = 1:10

bandnames = {'Beethoven','Vince Staples','grandson'};
train = 9; % samples to train
test = 5; % samples to confirm
sl = 1; % sample length in seconds

[Xtrain, ltrain, Xtest, ltest] = generatesamples(D,bandnames,train,test,sl,mp3Fs);

Xtrain = fft(Xtrain, [], 1);
Xtest = fft(Xtest, [], 1);

X = [Xtrain Xtest];

%% SVD
[u, s, v] = svd(X - mean(X(:)), 'econ');

PCs =[1 3 6]; % PC 3 for feature discrimination
%Xp = X';
%Y(:, PCs) = u(:,:)*Xp(:,PCs);

xtrain = v(1:size(Xtrain,2), PCs)'; %v(PCs, 1:size(Xtrain,2));
xtest = v(size(Xtrain,2)+1:end, PCs)'; %v(PCs, size(Xtrain,2)+1:end);

Model = fitcnb(real(xtrain'),ltrain);
test_labels = predict(Model,real(xtest'));
t1nbE = 100-sum((1/length(bandnames))*abs(test_labels-truth))/(length(bandnames)*test)*100;

trials(i) = t1bnE;

end

bar(trials)
yline(mean(trials),'--r', {sprintf('average = %.2f', mean(trials))})

xlabel('trial')
ylabel('accuracy')
title('Accuracy of Naive Bayes in 100 Trials')