 % Clear the Console
clc

% Load in the Images
disp("The input images have loaded."); 
imds = imageDatastore('./Resized_Input_Images',...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

T = imds.Labels;
T_size = size(T);

imdsTrimmed = splitEachLabel(imds, 100, 'randomized');
imdsTrimmed.ReadFcn = @(filename)readAndPreprocessImageA(filename);
[imdsTrainA,imdsTestA] = splitEachLabel(imdsTrimmed,0.6,'randomized');

% Read the Train and Test Images
X = cell(imdsTrainA.readall)';
Y = cell(imdsTestA.readall)';

% Convert data to type double
for i = 1:length(X)
   X{i} = im2double(X{i});
end

for i = 1:length(Y)
    Y{i} = im2double(Y{i});
end

% Train the Autoencoder
disp("The Autoencoder is now running.");
hidden_size = 250;
autoenc = trainAutoencoder(X, hidden_size, ...
        'L2WeightRegularization',0.0001,...
        'useGPU', true, ...
        'MaxEpochs', 10000);
disp("The Autoencoder has finished running.");

% Reconstruct the images
features = predict(autoenc, Y);  

% Plot Original Images
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(X{i});
end

% Plot Reconstructed Images
figure;
for i = 1:100
    subplot(10 ,10, i);
    imshow(features{i});
end

% Save the Reconstructed Images to the output folder
outfolder = '/home/titan/Desktop/VAE_Reconstructed_Images/';
VAE_Image_Count = length(features);

disp("Writing the Reconstructed Images from the VAE to folder " + outfolder);
%for img_num = 1:VAE_Image_Count
%    img_name = "VAE_"+ num2str(img_num) + '.jpg';
%    disp(img_name + " was written to the folder.");
%    filename = fullfile(outfolder, img_name);
%    imwrite(features{i}, filename, 'jpg');  
%end
%disp("The images were successfully written.");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-process the Images 
function Iout = readAndPreprocessImageA(filename)
        I = imread(filename);
        
        if ismatrix(I)
            I = cat(3,I,I,I);
        end
        
        %gray_img = rgb2gray(I);
        Iout = imresize(I, [227 227]);
end