%% for our data

emotionsUsed=[0,1,2,3,4,5,6,7];
[imagesData shapeData labels stringLabels] = extractData('CKDB', emotionsUsed);
imagesData=reshape(imagesData,size(imagesData,1),size(imagesData,2),size(imagesData,3),1);
imagesData = permute(imagesData,[2 3 4 1]);
labels=permute(labels,[2 1]);
labels=categorical(labels);
imagesData=imresize(imagesData,[84,84]);


indexes = crossvalind('Kfold',size(imagesData,4),6);
k=1;
Xtrain = imagesData(:,:,:,indexes~=k);   %4D   
Ytrain  = labels(indexes~=k,:);  %530 images
Xtest  =   imagesData(:,:,:,indexes==k);  %4D 
Ytest   = labels(indexes==k,:); %106 images

%%
layers = [
    imageInputLayer([84 84 1])

    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(6,32,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(14,64,'Padding',3)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(8)
    softmaxLayer
    classificationLayer];

%%
frequency=4;
options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...
    'ValidationData',{Xtest,Ytest},...
    'ValidationFrequency',frequency,...
    'Verbose',false,...
    'Plots','training-progress');
%valDigitData= [Xtest,Ytest]
%%
net = trainNetwork(Xtrain,Ytrain,layers,options);
%trainDigitData = [Xtrain,Ytrain]
%%
predictedLabels = classify(net,Xtest);
valLabels = Ytest;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels)

%% plot of the weights for the first convolutional layer
 w1=net.Layers(2).Weights;
 w1 = mat2gray(w1);
 w1 = imresize(w1,20); 
 
 figure
 montage(w1)
 title('First convolutional layer weights')
