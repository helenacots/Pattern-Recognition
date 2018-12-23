function [ accuracy confusionMatrix ] = applyMethods(data, labels, labelsUsed, indexesCrossVal, classificationMethod,dimensionalityReductionMethod,dim)
    
    % ApplyMethods
    
    % This function estimates the accuracy and confusion over the dataset samples in data by using Cross Validation 
    % the input template matching method and the error measure method.
    
    % INPUTS:
    % data: NxDxD , matrix with the images or shape data of the dataset
    % labels: 1xN vector with the emotion labels for each sample in the matrix data.
    % labelsUsed: labels used to train and classify.
    % templateMethod: string with the method used to generate the template.
    % errorMeasuse: string with the error measure method used to compare the template with the samples.
    % indexesCrossVal: indexes used to perform the performance evaluation with cross validation

    indexes = indexesCrossVal;
    K = max(indexes);

    confusionMatrix = zeros(numel(labelsUsed));
    
    if strcmp(classificationMethod,'Mahalanobis')
        if strcmp(dimensionalityReductionMethod,'none')
        dimensionalityReductionMethod='PCA';
        end
        dim=13;
    end

    for k = 1:K
        display(['Testing data subset: ' num2str(k) '/' num2str(K)]);
        %get train and test dataset with the indexes obtained with the KFold
        %cross validation
        trainSamples = data(indexes~=k,:,:);
        labelsTrain  = labels(indexes~=k);
        
        testSamples  = data(indexes==k,:,:);
        labelsTest   = labels(indexes==k);

        switch dimensionalityReductionMethod
            case 'PCA' %dimensionality reduction with PCA
            [ trainSamples, meanProjectionPCA, vectorsProjectionPCA ] = reduceDimensionality(trainSamples, 'PCA', dim, labelsTrain);
            [ testSamples ] = projectData(testSamples, meanProjectionPCA,vectorsProjectionPCA);
            case 'LDA' %dimensionality reduction with LDA
            [ trainPCA, meanProjectionPCA, vectorsProjectionPCA ] = reduceDimensionality(trainSamples, 'PCA', 500, labelsTrain);
            [ testPCA ] = projectData(testSamples,meanProjectionPCA, vectorsProjectionPCA);
            [ trainSamples, meanProjectionLDA, vectorsProjectionLDA ] = reduceDimensionality(trainPCA, 'LDA', dim, labelsTrain);
            [ testSamples ] = projectData(testPCA,meanProjectionLDA, vectorsProjectionLDA);

        end

     
         switch classificationMethod
            case 'example'
                % SAMPLE OF MATLAB's implementation of several classifiers
                knn = fitcknn(trainSamples, labelsTrain);
                estimatedLabels=knn.predict(testSamples);
                
            case 'SVM' %Accuracy increments
                % TODO:
                % Train and classify with an implementation of SVM
                % HINT: check Matlab's svmtrain / svmclassify
                % REMEMBER: basic SVM is intended for binary classification. It MUST be extended to a 
                %  multiclass level, look for a strategy (data partition, iterative one-against-all, etc)
                svm=fitcecoc(trainSamples,labelsTrain);%does a binary classification for the classes 
                estimatedLabels=svm.predict(testSamples);
                
            case 'Mahalanobis' %hem de reduir les dimensions
                c= 1;
                for e = labelsUsed
                trainEmotion = trainSamples(labelsTrain==e,:);
                dist(c,:)=mahal(testSamples,trainEmotion);
                c = c+1;
                end
                for i=1:size(dist,2)
                    estimatedLabels(i) = labelsUsed(find(dist(:,i)==min(dist(:,i)),1));
                end
           
            case 'kernelSVM'
                %searching in Matlab templateSVM 
                kernel='polynomial'; % 'gaussian' or 'polynomial'
                t=templateSVM('KernelFunction',kernel,'KernelScale','auto','Standardize',1);
                ksvm=fitcecoc(trainSamples,labelsTrain,'Learners',t,'FitPosterior',1);
                estimatedLabels=ksvm.predict(testSamples);     
        end

        %Create confusion matrix evaluating the templates with the test data
        confusionMatrix = confusionMatrix + confusionmat(estimatedLabels, labelsTest, 'ORDER', labelsUsed);
    end
    
    %get the total accuracy of the system
    accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
end

