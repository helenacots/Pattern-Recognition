function [ h ] = drawShape( shape )
    %DRAWSHAPE Summary of this function goes here
    %   Detailed explanation goes here
    h = plot(shape(:,1), shape(:,2),'r*');
    hold on;
    contourPoints = 1:17;
    plot(shape(contourPoints,1),shape(contourPoints,2),'-b');
    leftEyePoints = [37:42 37];
    plot(shape(leftEyePoints ,1),shape(leftEyePoints ,2),'-b');
    leftRightPoints = 43:48;
    plot(shape(leftRightPoints,1),shape(leftRightPoints,2),'-b');
    %eyebrows
    plot(shape([18:22 18],1),shape([18:22 18],2),'-b');
    plot(shape([23:27 23],1),shape([23:27 23],2),'-b');
    %nose 
    plot(shape([28:31 28],1),shape([28:31 28],2),'-b');
    plot(shape([32:36 32],1),shape([32:36 32],2),'-b');
    %mouth
    plot(shape([49:60 49],1),shape([49:60 49],2),'-b');
    plot(shape([61:68 61],1),shape([61:68 61],2),'-b');

end

