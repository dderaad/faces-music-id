close all; clear variables; clc

%% Directory
% https://ytmp3.cc/

mp3Fs = 44100; % Sample Rate, Hz

mdir = 'music';
D = dir(mdir);
D = D(3:end);

%% (test 1) Different Genre Bands
% Can we guess whether a clip is from one of the following different bands:
% Beethoven - classical
% clipping - experimental hip-hop
% grandson - alt-rock

bandnames = {'Beethoven','clipping','grandson'};
train = 60; % samples to train
test = 10; % samples to confirm
sl = 5; % sample length in seconds

tic
[Xtrain, ltrain, Xtest, ltest] = generatesamples(D,bandnames,train,test,sl,mp3Fs);
toc

Xtrain = fft(Xtrain, [], 1);
Xtest = fft(Xtest, [], 1);

X = [Xtrain Xtest];

%% SVD
[u, s, v] = svd(X - mean(X(:)), 'econ');

%% Classification
fig1 = figure(1);
plot(diag(s))
title('Singular Value Spectrum for Test 1')
ylabel('Singular Value')
xlabel('Mode')

% Y = u*X'; this creates a matrix that is big and useless
% We will form the first PCs columns of this matrix
% C(i,j) = A(i,:)*B(:,j)

PCs =[1 3 6]; % PC 3 for feature discrimination
%Xp = X';
%Y(:, PCs) = u(:,:)*Xp(:,PCs);

xtrain = v(1:size(Xtrain,2), PCs)'; %v(PCs, 1:size(Xtrain,2));
xtest = v(size(Xtrain,2)+1:end, PCs)'; %v(PCs, size(Xtrain,2)+1:end);

fig2 = figure(2);
sgtitle('Samples as Points in Real and Imaginary Principal Component Space')
subplot(2,1,1)
scatter3(real(xtrain(1,ltrain==1)), real(xtrain(2,ltrain==1)), real(xtrain(3,ltrain==1)), 10, [1,0,0])
hold on
scatter3(real(xtrain(1,ltrain==2)), real(xtrain(2,ltrain==2)), real(xtrain(3,ltrain==2)), 10, [0,1,0])
scatter3(real(xtrain(1,ltrain==3)), real(xtrain(2,ltrain==3)), real(xtrain(3,ltrain==3)), 10, [0,0,1])
title('Real')
xlabel(sprintf('PC %d', PCs(1)))
ylabel(sprintf('PC %d', PCs(2)))
zlabel(sprintf('PC %d', PCs(3)))
legend(bandnames)
subplot(2,1,2)
scatter3(imag(xtrain(1,ltrain==1)), imag(xtrain(2,ltrain==1)), imag(xtrain(3,ltrain==1)), 10, [1,0,0])
hold on
scatter3(imag(xtrain(1,ltrain==2)), imag(xtrain(2,ltrain==2)), imag(xtrain(3,ltrain==2)), 10, [0,1,0])
scatter3(imag(xtrain(1,ltrain==3)), imag(xtrain(2,ltrain==3)), imag(xtrain(3,ltrain==3)), 10, [0,0,1])
title('Imaginary')
xlabel(sprintf('PC %d', PCs(1)))
ylabel(sprintf('PC %d', PCs(2)))
zlabel(sprintf('PC %d', PCs(3)))
legend(bandnames)

lclass = classify(real(xtest'),real(xtrain'),ltrain);
qclass = classify(real(xtest'),real(xtrain'),ltrain,'quadratic');
truth = ltest;
t1lE = 100-sum((1/length(bandnames))*abs(lclass-truth))/(length(bandnames)*test)*100;
t1qE = 100-sum((1/length(bandnames))*abs(qclass-truth))/(length(bandnames)*test)*100;

Model = fitcnb(real(xtrain'),ltrain);
test_labels = predict(Model,real(xtest'));
t1nbE = 100-sum((1/length(bandnames))*abs(test_labels-truth))/(length(bandnames)*test)*100;


%% (test 2) Same Genre Bands
% Can we guess whether a clip is from one of the following different bands
% of the same genre:
% clipping - experimental hip-hop
% Kendrick Lamar - hip-hop
% Vince Staples - hip-hop

bandnames = {'clipping','Kendrick Lamar','Vince Staples'};
train = 60; % samples to train
test = 10; % samples to confirm
sl = 5; % sample length in seconds
[Xtrain, ltrain, Xtest, ltest] = generatesamples(D,bandnames,train,test,sl,mp3Fs);

Xtrain = fft(Xtrain, [], 1);
Xtest = fft(Xtest, [], 1);

X = [Xtrain Xtest];

%% SVD
[u, s, v] = svd(X - mean(X(:)), 'econ');

%% Classification
fig3 = figure(3);
plot(diag(s))
title('Singular Value Spectrum for Test 2')
ylabel('Singular Value')
xlabel('Mode')

% Y = u*X'; this creates a matrix that is big and useless
% We will form the first PCs columns of this matrix
% C(i,j) = A(i,:)*B(:,j)

PCs = [1 2 3 5]; % PC 3 for feature discrimination
Xp = X';
%Y(:, PCs) = u(:,:)*Xp(:,PCs);

xtrain = v(1:size(Xtrain,2), PCs)'; %v(PCs, 1:size(Xtrain,2));
xtest = v(size(Xtrain,2)+1:end, PCs)'; %v(PCs, size(Xtrain,2)+1:end);

fig4 = figure(4);
sgtitle('Samples as Points in Real and Imaginary Principal Component Space')
subplot(2,1,1)
scatter3(real(xtrain(1,ltrain==1)), real(xtrain(2,ltrain==1)), real(xtrain(3,ltrain==1)), 10, [1,0,0])
hold on
scatter3(real(xtrain(1,ltrain==2)), real(xtrain(2,ltrain==2)), real(xtrain(3,ltrain==2)), 10, [0,1,0])
scatter3(real(xtrain(1,ltrain==3)), real(xtrain(2,ltrain==3)), real(xtrain(3,ltrain==3)), 10, [0,0,1])
title('Real')
xlabel(sprintf('PC %d', PCs(1)))
ylabel(sprintf('PC %d', PCs(2)))
zlabel(sprintf('PC %d', PCs(3)))
legend(bandnames)
subplot(2,1,2)
scatter3(imag(xtrain(1,ltrain==1)), imag(xtrain(2,ltrain==1)), imag(xtrain(3,ltrain==1)), 10, [1,0,0])
hold on
scatter3(imag(xtrain(1,ltrain==2)), imag(xtrain(2,ltrain==2)), imag(xtrain(3,ltrain==2)), 10, [0,1,0])
scatter3(imag(xtrain(1,ltrain==3)), imag(xtrain(2,ltrain==3)), imag(xtrain(3,ltrain==3)), 10, [0,0,1])
title('Imaginary')
xlabel(sprintf('PC %d', PCs(1)))
ylabel(sprintf('PC %d', PCs(2)))
zlabel(sprintf('PC %d', PCs(3)))
legend(bandnames)

lclass = classify(real(xtest'),real(xtrain'),ltrain);
qclass = classify(real(xtest'),real(xtrain'),ltrain,'quadratic');
truth = ltest;
t2lE = 100-sum((1/length(bandnames))*abs(lclass-truth))/(length(bandnames)*test)*100;
t2qE = 100-sum((1/length(bandnames))*abs(qclass-truth))/(length(bandnames)*test)*100;

Model = fitcnb(real(xtrain'),ltrain);
test_labels = predict(Model,real(xtest'));
t2nbE = 100-sum((1/length(bandnames))*abs(test_labels-truth))/(length(bandnames)*test)*100;


%% (test 3) Genre Classification
% Can we categorize music as one of the following genres:
% hip-hop
% alt-rock
% classical

bandnames = {'hip-hop','alt-rock','classical'};
train = 60; % samples to train
test = 10; % samples to confirm
sl = 5; % sample length in seconds
[Xtrain, ltrain, Xtest, ltest] = generatesamples(D,bandnames,train,test,sl,mp3Fs);

Xtrain = fft(Xtrain, [], 1);
Xtest = fft(Xtest, [], 1);

X = [Xtrain Xtest];

%% SVD
[u, s, v] = svd(X - mean(X(:)), 'econ');

%% Classification
fig5 = figure(5);
plot(diag(s))
title('Singular Value Spectrum for Test 3')
ylabel('Singular Value')
xlabel('Mode')

% Y = u*X'; this creates a matrix that is big and useless
% We will form the first PCs columns of this matrix
% C(i,j) = A(i,:)*B(:,j)

PCs = [1 2 3 5]; % PC 3 for feature discrimination
Xp = X';
%Y(:, PCs) = u(:,:)*Xp(:,PCs);

xtrain = v(1:size(Xtrain,2), PCs)'; %v(PCs, 1:size(Xtrain,2));
xtest = v(size(Xtrain,2)+1:end, PCs)'; %v(PCs, size(Xtrain,2)+1:end);

fig6 = figure(6);
sgtitle('Samples as Points in Real and Imaginary Principal Component Space')
subplot(2,1,1)
scatter3(real(xtrain(1,ltrain==1)), real(xtrain(2,ltrain==1)), real(xtrain(3,ltrain==1)), 10, [1,0,0])
hold on
scatter3(real(xtrain(1,ltrain==2)), real(xtrain(2,ltrain==2)), real(xtrain(3,ltrain==2)), 10, [0,1,0])
scatter3(real(xtrain(1,ltrain==3)), real(xtrain(2,ltrain==3)), real(xtrain(3,ltrain==3)), 10, [0,0,1])
title('Real')
xlabel(sprintf('PC %d', PCs(1)))
ylabel(sprintf('PC %d', PCs(2)))
zlabel(sprintf('PC %d', PCs(3)))
legend(bandnames)
subplot(2,1,2)
scatter3(imag(xtrain(1,ltrain==1)), imag(xtrain(2,ltrain==1)), imag(xtrain(3,ltrain==1)), 10, [1,0,0])
hold on
scatter3(imag(xtrain(1,ltrain==2)), imag(xtrain(2,ltrain==2)), imag(xtrain(3,ltrain==2)), 10, [0,1,0])
scatter3(imag(xtrain(1,ltrain==3)), imag(xtrain(2,ltrain==3)), imag(xtrain(3,ltrain==3)), 10, [0,0,1])
title('Imaginary')
xlabel(sprintf('PC %d', PCs(1)))
ylabel(sprintf('PC %d', PCs(2)))
zlabel(sprintf('PC %d', PCs(3)))
legend(bandnames)

lclass = classify(real(xtest'),real(xtrain'),ltrain);
qclass = classify(real(xtest'),real(xtrain'),ltrain,'quadratic');
truth = ltest;
t3lE = 100-sum((1/length(bandnames))*abs(lclass-truth))/(length(bandnames)*test)*100;
t3qE = 100-sum((1/length(bandnames))*abs(qclass-truth))/(length(bandnames)*test)*100;

Model = fitcnb(real(xtrain'),ltrain);
test_labels = predict(Model,real(xtest'));
t3nbE = 100-sum((1/length(bandnames))*abs(test_labels-truth))/(length(bandnames)*test)*100;


%% Results
fmt = 'test %d \n LDM: %.2f \n QDM: %.2f \n NB: %.2f';
result1 = sprintf(fmt, 1, t1lE, t1qE, t1nbE);
result2 = sprintf(fmt, 2, t2lE, t2qE, t2nbE);
result3 = sprintf(fmt, 3, t3lE, t3qE, t3nbE);

disp(result1)
disp(result2)
disp(result3)

%% Saving

saveas(fig1,'svtest1.png')
saveas(fig2,'PCmusic1.png')
saveas(fig3,'svtest2.png')
saveas(fig4,'PCmusic2.png')
saveas(fig5,'svtest3.png')
saveas(fig6,'PCmusic3.png')