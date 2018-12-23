function [ pattern ] = createTemplate( data, namePattern )
%CREATETEMPLATE Given the samples in the data matrix, create a template
%using the namePattern method. 
     %squeeze treu la "singleton dimention " d'un array (1x2x2-->2x2)
    switch namePattern
        case 'grayscaleMean'
            %mean of the grayscale images
            pattern = squeeze(mean(data));
        case 'grayscaleMedian'
            %mediana de les imatges 
            pattern = squeeze(median(data));
        case 'meanShape'
            %mitja de les matrius tipu shapeData
            pattern = squeeze(mean(data));
        case 'gaussian'
            data = imgaussfilt(data,0.5); %apliquem filtre gaussià (imatge + borrosa)
            pattern = squeeze(mean(data)); %fem la mitja després del filtre
        case 'noiseFilter'
            %apliquem filtre per treure soroll de les imatges
             sum = zeros(128);
             for i=1:size(data,1)
                 newdata = squeeze(data(i,:,:));
                 dataWithoutNoise = wiener2(newdata,[size(newdata,1) size(newdata,2)]);
                 sum = sum + dataWithoutNoise;
             end
             %després de treure el soroll, fem la mitja de la nova data
             noiseRemoved = sum/(size(data,1));
             pattern = noiseRemoved;    
         case 'mouth'
            %fem la mitja dels píxels de la boca 
            pattern = squeeze(mean(data(:,60:100,30:90))); 
        case 'double'
            %fem la mitja de les imatges normalitzades
            pattern = squeeze(mean(im2double(data)));
    end
end

% aqui hem d'afegir més "case"s templates per a les mostres 