% Fusion Script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc 
clear


disp("Loading the data.");
imds = imageDatastore('./FusionCode/FullSet/Original',...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imds = shuffle(imds);

tbl = countEachLabel(imds);

[train,test] = splitEachLabel(imds,0.6,'randomized');


% numImgs = 40;


% Alexnet and vgg16/19 use different sized input images.
% Use splitEachLabel method to trim two sets, for Alexnet and VGG based nets.
train.ReadFcn = @(filename)readAndPreprocessImageA(filename);
trainA = train;
test.ReadFcn = @(filename)readAndPreprocessImageA(filename);


testA = test;

train2.ReadFcn = @(filename)readAndPreprocessImageV(filename);
trainV = trainA;
test2.ReadFcn = @(filename)readAndPreprocessImageV(filename);
testV = testA;
% select three pretrained networks
net1 = alexnet;
net2 = vgg16;
net3 = vgg19;

disp("Extracting Features");

% set layer to fc7
layer = 'fc7';

% extract training features from 3 pretrained networks
featuresTrainA = activations(net1,trainA,layer,...
    'MiniBatchSize', 1,'OutputAs','rows',...
    'ExecutionEnvironment','gpu');
trainV = trainA;
train.ReadFcn = @(filename)readAndPreprocessImageV(filename);


featuresTrainB = activations(net2,trainV,layer,...
    'MiniBatchSize', 1,'OutputAs','rows',...
    'ExecutionEnvironment','gpu');

featuresTrainC = activations(net3,trainV,layer,...
    'MiniBatchSize', 1,'OutputAs','rows',...
    'ExecutionEnvironment','gpu');

featuresFused = featuresTrainA + featuresTrainB + featuresTrainC;

[TrainM, TrainN] = size(featuresTrainA);

for(i = 1:numel(featuresTrainA))
    featuresMax(i) = max(featuresTrainA(i), featuresTrainB(i));
    featuresMax(i) = max(featuresMax(i), featuresTrainC(i));
end

featuresMax = reshape(featuresMax, [TrainM, TrainN]);


for(i = 1:numel(featuresTrainA))
    featuresMin(i) = min(featuresTrainA(i), featuresTrainB(i));
    featuresMin(i) = min(featuresMin(i), featuresTrainC(i));
end


featuresMin = reshape(featuresMin, [TrainM, TrainN]);

for(i = 1:numel(featuresTrainA))
    featuresAvg(i) = (featuresTrainA(i) + featuresTrainB(i) + featuresTrainC(i) ) / 3;
end
featuresAvg = reshape(featuresAvg, [TrainM, TrainN]);

featuresTestA = activations(net1,testA,layer,...
    'MiniBatchSize', 1,'OutputAs','rows',...
    'ExecutionEnvironment','gpu');
testV = testA;
test.ReadFcn = @(filename)readAndPreprocessImageV(filename);

featuresTestV1 = activations(net2,testV,layer,...
    'MiniBatchSize', 1,'OutputAs','rows',...
    'ExecutionEnvironment','gpu');

featuresTestV2 = activations(net3,testV,layer,...
    'MiniBatchSize', 1,'OutputAs','rows',...
    'ExecutionEnvironment','gpu');

featuresTestFused = featuresTestA + featuresTestV1 + featuresTestV2;

[TestM, TestN] = size(featuresTestA);

for(i = 1:numel(featuresTestA))
    featuresTestMax(i) = max(featuresTestA(i), featuresTestV1(i));
    featuresTestMax(i) = max(featuresTestMax(i), featuresTestV2(i));
end

featuresTestMax = reshape(featuresTestMax, [TestM, TestN]);

for(i = 1:numel(featuresTestA))
    featuresTestMin(i) = min(featuresTestA(i), featuresTestV1(i));
    featuresTestMin(i) = min(featuresTestMin(i), featuresTestV2(i));
end

featuresTestMin = reshape(featuresTestMin,[TestM, TestN]);

for(i = 1:numel(featuresTestA))
    featuresTestAvg(i) = (featuresTestA(i) + featuresTestV1(i) + featuresTestV2(i) ) / 3;
end

featuresTestAvg = reshape(featuresTestAvg,[TestM, TestN]);


YTrain = trainA.Labels;
YTest = testA.Labels;

disp("Training SVM from Features");


classifier = fitcecoc(featuresFused,YTrain,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');

classifierMax = fitcecoc(featuresMax,YTrain,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
classifierMin = fitcecoc(featuresMin,YTrain,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
classifierAvg = fitcecoc(featuresAvg,YTrain,'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');

disp("Predictions:GAN&VAE");
disp("Alexnet Features:");
tic
YPred = predict(classifier,featuresTestA);
anTestTime = toc;
accuracy1 = mean(YPred == YTest)

disp("VGG16 Features:");
tic
YPred = predict(classifier,featuresTestV1);
Time = toc;
accuracy2 = mean(YPred == YTest)

disp("VGG19 Features:");
YPred = predict(classifier,featuresTestV2);
accuracy3 = mean(YPred == YTest)

disp("Fused Features(Sum):");
YPred = predict(classifier,featuresTestFused);
accuracySum = mean(YPred == YTest)

disp("Fused Features(Max):");
YPred = predict(classifier,featuresTestMax);
accuracyMax = mean(YPred == YTest)

disp("Fused Features(Min):");
YPred = predict(classifier,featuresTestMin);
accuracyMin = mean(YPred == YTest)

disp("Fused Features(Avg):");
YPred = predict(classifier,featuresTestAvg);
accuracyAvg = mean(YPred == YTest)


accuracy = [accuracy1, accuracy2, accuracy3];
dlmwrite('Results.csv',accuracy,'delimiter',',','-append');
fusionAccuracy = [accuracySum, accuracyMax, accuracyMin, accuracyAvg];
dlmwrite('FusionResults.csv',fusionAccuracy,'delimiter',',','-append');

function Iout = readAndPreprocessImageA(filename)

        I = imread(filename);

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I)
            I = cat(3,I,I,I);
        end

        % Resize the image as required for Alexnet.
        Iout = imresize(I, [227 227]);

        % Note that the aspect ratio is not preserved.
end

function Iout = readAndPreprocessImageV(filename)

        I = imread(filename);

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I)
            I = cat(3,I,I,I);
        end

        % Resize the image as required for VGG.
        Iout = imresize(I, [224 224]);

        % Note that the aspect ratio is not preserved.
end
