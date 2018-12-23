function [ imagesData shapeData labels ] = extractData( databaseDirectory , emotionsUsed )
%EXTRACTDATA Given the directory of the database. This function read all
%the emotions in the database and return 3 variables.
%imagesData: contains a Nx128x128 with the N images of the database
%shapeData: contains a Nx68x2 with the N shapes of the database
%labels: contains the number of emotion corresponding to each sample in
%the database

    emotions = emotionsUsed;

    %compute the total number of images in the DB
    totalNumberImages = 0;
    for numEmotion = emotions
        pathEmotion = [databaseDirectory '/' num2str(numEmotion)]; 
        files = ListFiles(pathEmotion);
        totalNumberImages = totalNumberImages + numel(files);
    end
    
    %Iterate over all the emotions
    imagesData = zeros(totalNumberImages,128,128);
    shapeData = zeros(totalNumberImages,68,2);
    labels = []; %etiquetes
    numberImage = 1;
    for numEmotion = emotions
        pathEmotion = [databaseDirectory '/' num2str(numEmotion)];    
        %List the images files
        files = ListFiles(pathEmotion);
        %Variables to store the data for each emotion image
        labels = [labels ones(1,numel(files))*numEmotion];
        for i = 1:numel(files)
            [path name extenstion] = fileparts(files(i).name); %returns the path, the filename and the file extension for the especific file
            pathFile = [pathEmotion '/' name ];
            %read image file
            image = imread([pathFile '.tiff']);
            %read shape file
            shape = dlmread([pathFile '.txt']);
            %store grayscale image
            imagesData(numberImage,:,:) = image;
            %store shape
            shapeData(numberImage,:,:) = shape;   
            numberImage = numberImage + 1;
        end
        
    end

end

