%The program represents the Lucy Richardson algorithm for image restoration.
%author: Aditya Pulekar

clc;
clear all;
close all;
img=imread('carlicense_noisy.png');
% psf=fspecial('gaussian',4,4);
figure;
title('Original Image')
imshow(img)
% afterMedFil=medfilt2(img,[3 3]);
% figure;
% imshow(afterMedFil);
% debl=img;
% for i=1:5
%   debl=deconvlucy(debl,psf,3);
% end
% debl=deconvwnr(img,psf,1);
psf=fspecial('gaussian',10,50);   %10,50
debl=deconvlucy(img,psf);
% psf=fspecial('gaussian',7,10);
% o=padarray(ones(size(psf)-4),[4 4],'replicate','both');
% [debl,p]=deconvblind(img,o);
figure;
title('Deblurred Image');
imshow(debl);
imwrite(debl,'DeblurredImage.jpg');