%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% P3 - RECONEIXEMENT DE PATRONS  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    M�TODES DE CLASSIFICACI�    %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
addpath(genpath('.'));

%choose the emotion labels we want to classify in the database
% 0:Neutral 
% 1:Angry 
% 2:Bored 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise
emotionsUsed = [0 1 2 3 4 5 6 7];  

%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
[imagesData shapeData labels stringLabels] = extractData('CKDB', emotionsUsed);

%%%%%%%%%%%%%%%% EXTRACT FEATURES %%%%%%%%%%%%
grayscaleFeatures = extractFeaturesFromData(imagesData,'grayscale');


%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);

classificationMethod='kernelSVM'; %SVM, kernel SVM, Mahalanobis
dimensionalityReductionMethod='LDA'; %PCA, LDA or 'none' if we don't want reduction
dim=200; %reduced dimensions 
%%%%%%%  EXAMPLE OF CLASSIFYING THE EXPRESSION USING TEMPLATE  MATCHING %%%%
[acuracy conf] = applyMethods(grayscaleFeatures, labels, emotionsUsed, indexesCrossVal, classificationMethod,dimensionalityReductionMethod,dim)


