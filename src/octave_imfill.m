#! /usr/local/bin/octave -qf
pkg load image
IMG = imread('../output/06dilated_image.jpg');
FILL = imfill(IMG);
imwrite(FILL, '../output/07filled_image.png')
