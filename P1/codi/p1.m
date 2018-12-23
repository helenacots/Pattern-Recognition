
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% P1 - RECONEIXEMENT DE PATRONS %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%    TEMPLATE MATCHING          %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%choose the emotion labels we want to classify in the database
% 0:Neutral 
% 1:Angry 
% 2:Bored 
% 3:Disgust 
% 4:Fear 
% 5:Happiness 
% 6:Sadness 
% 7:Surprise
emotionsUsed = [0 1 3 4 5 6 7];  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% EXTRACT DATA %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[imagesData shapeData labels] = extractData('../CKDB', emotionsUsed);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% DIVIDE DATA (TRAIN/TEST) WITH CROSS VALIDATION  %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%per a obtenir una mitja de diferents "accuracys":
accString = [];
for i = 0:5
K = 3;
indexesCrossVal = crossvalind('Kfold',size(imagesData,1),K);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% TEST DIFFERENT TEMPLATES METHODS %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = imagesData; %or shapeData
%si volem utilitzar el mètode 'meanShape' --> data= shapeData
    [accuracy confMatrix] = testMethod(data , labels, emotionsUsed ,  'noiseFilter', 'hamming', indexesCrossVal);
    accString = [accString, accuracy];
end
ACCURACYMEAN = mean(accString)


