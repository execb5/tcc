#! /usr/local/bin/octave -qf
pkg load image
IMG = imread('dilated_image.jpg');
FILL = imfill(IMG);
imwrite(FILL, 'filled_image.png')
