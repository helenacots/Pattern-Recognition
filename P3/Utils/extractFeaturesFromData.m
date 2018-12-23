function [ features ] = extractFeaturesFromData( data , featureType )
%EXCTRACTFEATURESFROMDATA Summary of this function goes here
%   Detailed explanation goes here
    switch featureType
        case 'grayscale'
            features = reshape(data,size(data,1),128*128);
        case 'shape'
            features = reshape(data,size(data,1),68*2);
    end
end

