function [ projectedData ] = projectData( data, meanProjection, projectionBasis )
%PROJECTDATA project the N points contained in the rows of the matrix data into the orthogonal projection defined 
%in the matrix projectionBasis. Previously, the data is translated using the vector meanProjection  
    mean = repmat(meanProjection,size(data,1),1);
    projectedData =  (projectionBasis'*(double(data) - mean)')';

end

