
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% P2 - RECONEIXEMENT DE PATRONS  %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    REDUCCIO DE DIMENSIONALITAT %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%% Ex.1
%extract the descriptor (intensities of pixels) --> done above 
%project them in 3D using PCA
[ dataProjected, meanProjection, vectorsProjection ] = reduceDimensionality(grayscaleFeatures, 'PCA', 3, labels);

%plot the data in the new 3D space
figure(1)
gscatter3(dataProjected(:,1),dataProjected(:,2),dataProjected(:,3), stringLabels, 8)
title('Data with PCA reduction')

%% Ex.2 
%plot the mean obtained with PCA
figure(1)
imagesc(reshape(meanProjection,[128,128]))
title('Mean obtained with PCA')

%plot the 3 bases of the projection matrix obtained
figure(2)
imagesc(reshape(vectorsProjection(:,1),[128,128]))
title('1st base of the projection matrix')

figure(3)
imagesc(reshape(vectorsProjection(:,2),[128,128]))
title('2nd base of the projection matrix')

figure(4)
imagesc(reshape(vectorsProjection(:,3),[128,128]))
title('3rd base of the projection matrix')

%% Ex 2 using surf
%plot the mean obtained with PCA
figure(1)
surf(reshape(meanProjection,[128,128]))
title('Mean obtained with PCA')

%plot the 3 bases of the projection matrix obtained
figure(2)
surf(reshape(vectorsProjection(:,1),[128,128]))
title('1st base of the projection matrix')

figure(3)
surf(reshape(vectorsProjection(:,2),[128,128]))
title('2nd base of the projection matrix')

figure(4)
surf(reshape(vectorsProjection(:,3),[128,128]))
title('3rd base of the projection matrix')

%% Ex.3

%Take an image of the DB 
image = grayscaleFeatures(24,:);
%plot the original image
subplot(1,2,1)
imagesc(reshape(image,128,128))
title('Original Image')
dim=100;
%reduce its dimensionality and reproject it in the original space
[ dataProjected, meanProjection, vectorsProjection ] = reduceDimensionality(grayscaleFeatures, 'PCA', dim, labels);
[ dataReprojected ] = reprojectData( dataProjected , meanProjection, vectorsProjection );
%plot the image with dimensionality reduced
subplot(1,2,2)
imagesc(reshape(dataReprojected(24,:), 128,128))
title(['Image dimensionally reduced to ' num2str(dim)])

%% repeat reducing the dimensionality to:
dims = [2,5,10,50,100,300,500];
ndims=size(dims,2);
image = grayscaleFeatures(24,:);
for i=1:ndims
    figure(i)
    subplot(1,2,1)
    imagesc(reshape(image,128,128))
    title('Original image')
    %reduce the dimensionality of the data (entire DB) with PCA
    [ dataProjected, meanProjection, vectorsProjection ] = reduceDimensionality(grayscaleFeatures, 'PCA', dims(i), labels);
    %reproject to the original space
    [ dataReprojected ] = reprojectData( dataProjected , meanProjection, vectorsProjection );
    subplot(1,2,2)
    imagesc(reshape(dataReprojected(24,:), 128,128))
    x=dims(i);
    title(['Image dimensionally reduced to ', num2str(x)])
end

%% Ex.4 
%same as 3, but reproducing the whole DB (done above)
dims = [2,5,10,50,100,300,500];
ndims = length(dims);
for i=1:ndims
    %reduce the dimensionality of the data (entire DB) with PCA
    [ dataProjected, meanProjection, vectorsProjection ] = reduceDimensionality(grayscaleFeatures, 'PCA', dims(i), labels);
    %reproject to the original space
    [ dataReprojected ] = reprojectData( dataProjected , meanProjection, vectorsProjection );
    %calculate quadratic error between reprojected and original data (for all of the images) 
    error(i)=sum(sum(((grayscaleFeatures-dataReprojected).^2),2)/size(dataReprojected,2),1)/size(dataReprojected,1);
end
%graph the evolution
plot(dims, error);
xlabel('Dimensions')
ylabel('Error')

%% Ex.5
%reduce the size of your data using PCA to the # of dims found
[ dataProjectedPCA, meanProjectionPCA, vectorsProjectionPCA ] = reduceDimensionality(grayscaleFeatures, 'PCA', 200, labels);
%reduce to 3 dimensions with LDA.
[ dataProjectedLDA, meanProjectionLDA, vectorsProjectionLDA ] = reduceDimensionality(dataProjectedPCA, 'LDA', 50, labels);
%plot the data in a 3D space
subplot(1,2,1)
gscatter3(dataProjectedPCA(:,1),dataProjectedPCA(:,2),dataProjectedPCA(:,3), stringLabels, 8)
title('PCA')
subplot(1,2,2)
gscatter3(dataProjectedLDA(:,1),dataProjectedLDA(:,2),dataProjectedLDA(:,3), stringLabels, 8)
title('LDA')
    
%% Ex.6
%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);
%dims = [2,5,10,50,100,300,500];
%apply any of the mothods u used in the previous practice for classifying
reduceMethod='LDA'; %'PCA' OR 'LDA'
dim=200;
errorMeasure='correlation';
[ accuracy confusionMatrix ] = testTemplateMatchingWithDR( grayscaleFeatures , labels, emotionsUsed , errorMeasure, indexesCrossVal,reduceMethod,dim )

%% PLOT of the accuracys to dimensionality reduction
dims = [2,5,10,50,100,300,500];
ndims=size(dims,2);

K = 6;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);
errorMeasure='correlation';
accuracyPCA=[];
accuracyLDA=[];
for i=1:ndims
[ accuracy confusionMatrix ] = testTemplateMatchingWithDR( grayscaleFeatures , labels, emotionsUsed , errorMeasure, indexesCrossVal,'PCA',dims(i) );
accuracyPCA=[accuracyPCA accuracy];
end
for i=1:ndims
errorMeasure='correlation';
[ accuracy confusionMatrix ] = testTemplateMatchingWithDR( grayscaleFeatures , labels, emotionsUsed , errorMeasure, indexesCrossVal,'LDA',dims(i) );
accuracyLDA=[accuracyLDA accuracy];
end

%graph the evolution
subplot(1,2,1)
plot(dims, accuracyPCA);
xlabel('Dimensions')
ylabel('Accuracy')
title('PCA')
axis([0 500 0 0.8])
subplot(1,2,2)
plot(dims, accuracyLDA);
xlabel('Dimensions')
ylabel('Accuracy')
title('LDA')
axis([0 500 0 0.8])