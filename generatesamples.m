function [Xtrain, ltrain, Xtest, ltest] = generatesamples(D,fldnames,train,test,samplelength,samplerate)
sl = samplelength;
Xtrain = zeros([sl*samplerate train*length(fldnames)]);
Xtest = zeros([sl*samplerate test*length(fldnames)]);
lr = ones([train 1]);
ltrain = [1*lr; 2*lr; 3*lr];
ls = ones([test 1]);
ltest = [1*ls; 2*ls; 3*ls];

bnd = -1;

for i = 1:length(D)
    in = isin(D(i).name, fldnames);
    if in
        bnd = bnd + 1;
        SD = dir(sprintf('%s\\%s',D(i).folder,D(i).name));
        SD = SD(3:end);
        
        trainsamples = train; 
        testsamples = test; 
        
        for j = 1:length(SD)
            % We will sample from each of the songs available at random.
            if trainsamples == 0 || j-length(SD)==0
                trainthissong = trainsamples;
            else
                trainthissong = randi(trainsamples);
            end
            
            if testsamples == 0 || j-length(SD)==0
                testthissong = testsamples;
            else
                testthissong = randi(testsamples);
            end
            
            trainsamples = trainsamples - trainthissong;
            testsamples = testsamples - testthissong;
            filename = sprintf('%s\\%s',SD(j).folder,SD(j).name);
            disp(filename)
            info = audioinfo(filename);
            %[y, Fs] = audioread(filename);
            %t = linspace(0, length(y)/Fs, length(y));
            fiveseconds = samplelength*samplerate;
            samplestarts = randperm(info.TotalSamples-fiveseconds-1, trainthissong + testthissong);
            trainstarts = samplestarts(1:trainthissong);
            teststarts = samplestarts(trainthissong+1:end);
            
            traintaken = train-trainsamples-trainthissong;
            testtaken = test-testsamples-testthissong;
            
            for k1 = 1:length(trainstarts)
                stereo = audioread(filename, [trainstarts(k1) trainstarts(k1)+fiveseconds-1]);
                Xindex = k1+bnd*train+traintaken;
                Xtrain(:,Xindex) = mono(stereo);
            end
            for k2 = 1:length(teststarts)
                stereo = audioread(filename, [teststarts(k2) teststarts(k2)+fiveseconds-1]);
                Xindex = k2+bnd*test+testtaken;
                Xtest(:,Xindex) = mono(stereo);
            end
        end
    end
end

end

