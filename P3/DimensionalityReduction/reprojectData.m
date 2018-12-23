function [ dataReprojected ] = reprojectData( dataProjected , meanProjection, matrixProjection )
%PROJECTDATA project the N points contained in the rows of the matrix data with inverse of the orthogonal projection defined 
%in the matrix projectionBasis. Then, the data is translated using the vector meanProjection  
    dataReprojected = (matrixProjection*dataProjected')' + repmat(meanProjection,size(dataProjected,1),1);
end

