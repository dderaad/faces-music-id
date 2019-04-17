close all; clear variables; clc

%% Load Cropped
% Load all cropped photos to put in X

n = 192;
m = 168;

crpdir = 'yalefaces_cropped\CroppedYale';
D = dir(crpdir);
D = D(3:end);

% The following chooses %numfolders% random folders and generates the faces
% from those only.
% numfolders = 4; % number of folders to grab pictures from
% 
% faces = D(randperm(length(D), numfolders));

numfolders = length(D);
faces = D;

X = zeros([n*m 64*numfolders]); % assuming 64 pictures in each folder

for i = 1:numfolders % for each folder of faces
    FD = dir(sprintf('%s\\%s',crpdir,faces(i).name));
    FD = FD(3:end);
    
    fld = (i-1)*length(FD);
    
    for j = 1:length(FD) % read faces into a column of X
        im = imread(sprintf('%s\\%s',FD(j).folder,FD(j).name));
        X(:,j+fld) = reshape(im, [n*m 1]);
    end
end

%% SVD on Cropped


[M,N]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
Xmn = repmat(mn,1,N);
X=X-Xmn; % subtract mean
[u,s,v] = svd(X/sqrt(N-1),'econ');

sv = diag(s);
var = sv.^2;
energy = var./sum(var);
energyperPC = cumsum(energy);
plotenergies = [.50 .90 .95 .99];

[rw,cl] = find(energyperPC>plotenergies);

x = [rw(find(cl==1, 1))
     rw(find(cl==2, 1)) 
     rw(find(cl==3, 1)) 
     rw(find(cl==4, 1))];


%% Plotting Cropped

fig1 = figure(1);
semilogx(energy, 'o', 'MarkerFaceColor', [0.75 0.75 1])
title(sprintf('Energy Captured by Each Singular Value \n Cropped Images'))
ylabel('Singular Value')
xlabel('Mode (Logarithmic Axis)')
hold on 
for i = 1:length(plotenergies)
    xline(x(i), '-', {sprintf('%.0f %%', 100*plotenergies(i))});
end
xticks([1 x' length(energy)])
axis tight

Y = u*X';

fig2 = figure(2);
colormap(bone)
a = 2; b = 4;
sgtitle(sprintf('The First %d Principle Components', a*b))
for k = 1:a*b
    subplot(a,b,k)
    imagesc(reshape(Y(k,:), [n m])), axis image
    title(sprintf('PC %d', k))
    xticks([])
    yticks([])
end

fig8 = figure(8);

colormap(bone)
a = 2; b = 4;
sgtitle(sprintf('The First %d Eigenfaces', a*b))
for k = 1:a*b
    subplot(a,b,k)
    imagesc(reshape(u(:,k), [n m])), axis image
    title(sprintf('Eigenface %d', k))
    xticks([])
    yticks([])
end

fig3 = figure(3);

r = x(plotenergies==.99);
sgtitle(sprintf('Rank r=%d Truncation', r))
sv(r+1:end) = 0;
srtrunc = diag(sv);

Xrtrunc = u*srtrunc*v.' * sqrt(N-1);

colormap(gray)
a = 3; b = 4; 
%b = numfolders; % randomized

% randomize which faces are
% printed
%faces2recon = sort(randperm(numfolders, a*b/2)); % randomized

faces2recon = [129 193 257 705 769 1025]; % specific faces for figure
for k = 1:2:a*b
    % conversion for randomized
    %imgi = ((faces2recon((k+1)/2)+1)/2-1)*64+1; 
    imgi = faces2recon((k+1)/2); % specific faces for figure\
    
    subplot(a,b,k)
    imagesc(reshape(X(:,imgi) + Xmn(:,imgi), [n m])), axis image
    title(sprintf('Original %d', imgi))
    
    xticks([])
    yticks([])
    
    subplot(a,b,k+1)
    imagesc(reshape(Xrtrunc(:,imgi) + Xmn(:,imgi), [n m])), axis image
    title(sprintf('Reproduction %d', imgi))
    
    xticks([])
    yticks([])
end



fig4 = figure(4);
colormap(gray)

imgids = [257 1153];
sgtitle(sprintf('Reproduction of Images %d, %d \n Under Different Rank-r Truncations', imgids(1), imgids(2)))

for p = 1:length(plotenergies)
    sv = diag(s);
    r = x(p);
    sv(r+1:end) = 0;
    srtrunc = diag(sv);

    Xrtrunc = u*srtrunc*v.' * sqrt(N-1);
    
    for id = 1:length(imgids)
        subplot(length(imgids),length(plotenergies),p+(id-1)*length(plotenergies))
        imagesc(reshape(Xrtrunc(:,imgids(id)) + Xmn(:,imgids(id)), [n m])), axis image
        title(sprintf('rank-r=%d \n %.0f %% energy', r, 100*plotenergies(p)))
        
        
        xticks([])
        yticks([])
    end
end


%% Load Uncropped

n = 243;
m = 320;

uncrpdir = 'yalefaces_uncropped\yalefaces';
D = dir(uncrpdir);
D = D(3:end);

% numfaces = 20; % random
% faces = D(randperm(length(D), numfaces)); % random

faces = D; % all faces

X = zeros([n*m length(faces)]);


for j = 1:length(faces) % read faces into a column of X
    im = imread(sprintf('%s\\%s',faces(j).folder,faces(j).name), 'gif');
    X(:,j) = reshape(im, [n*m 1]);
end


%% SVD on Uncropped

[M,N]=size(X); % compute data size
mn=mean(X,2); % compute mean for each row
Xmn = repmat(mn,1,N);
X=X-Xmn; % subtract mean

[u,s,v] = svd(X/sqrt(N-1),'econ');


sv = diag(s);
var = sv.^2;
energy = var./sum(var);
energyperPC = cumsum(energy);
plotenergies = [.50 .90 .95 .99];

[rw,cl] = find(energyperPC>plotenergies); 

x = [rw(find(cl==1, 1))
     rw(find(cl==2, 1)) 
     rw(find(cl==3, 1)) 
     rw(find(cl==4, 1))];
 
%% Plotting Uncropped

fig5 = figure(5);
plot(energy, 'o', 'MarkerFaceColor', [0.75 0.75 1])
title(sprintf('Energy Captured by Each Singular Value \n Uncropped Images'))
ylabel('Singular Value')
xlabel('Mode')
hold on 
for i = 1:length(plotenergies)
    xline(x(i), '-', {sprintf('%.0f %%', 100*plotenergies(i))});
end
xticks([0 x' length(energy)])
axis tight

fig6 = figure(6);

r = x(plotenergies==.99);
sgtitle(sprintf('Rank r=%d Truncation', r))
sv(r+1:end) = 0;
srtrunc = diag(sv);

Xrtrunc = u*srtrunc*v.' * sqrt(N-1);

colormap(gray)
a = 3; b = 4; 
%b = numfolders; % randomized

% randomize which faces are
% printed
%faces2recon = sort(randperm(length(D), a*b/2)); % randomized

faces2recon = [61 100 117 129 134 165]; % specific faces for figure
for k = 1:2:a*b
    % conversion for randomized
    imgi = faces2recon((k+1)/2); 
    %imgi = faces2recon((k+1)/2); % specific faces for figure\
    
    subplot(a,b,k)
    imagesc(reshape(X(:,imgi) + Xmn(:,imgi), [n m])), axis image
    title(sprintf('Original %d', imgi))
    
    xticks([])
    yticks([])
    
    subplot(a,b,k+1)
    imagesc(reshape(Xrtrunc(:,imgi) + Xmn(:,imgi), [n m])), axis image
    title(sprintf('Reproduction %d', imgi))
    
    xticks([])
    yticks([])
end


fig7 = figure(7);
colormap(gray)

imgids = [114 156 157];
sgtitle(sprintf(...
'Reproduction of Images %d, %d, %d \n Under Different Rank-r Truncations',...
imgids(1), imgids(2), imgids(3)))

for p = 1:length(plotenergies)
    sv = diag(s);
    r = x(p);
    sv(r+1:end) = 0;
    srtrunc = diag(sv);

    Xrtrunc = u*srtrunc*v.' * sqrt(N-1);
    
    for id = 1:length(imgids)
        subplot(length(imgids),length(plotenergies),p+(id-1)*length(plotenergies))
        imagesc(reshape(Xrtrunc(:,imgids(id)) + Xmn(:,imgids(id)), [n m])), axis image
        title(sprintf('rank-r=%d \n %.0f %% energy', r, 100*plotenergies(p)))
        
        
        xticks([])
        yticks([])
    end
end

%% Saving

saveas(fig1,'svface1.png')
saveas(fig2,'PC1.png')
saveas(fig8,'PC2.png')
saveas(fig3,'recon1.png')
saveas(fig4,'recon2.png')
saveas(fig5,'svface2.png')
saveas(fig6,'recon3.png')
saveas(fig7,'recon4.png')