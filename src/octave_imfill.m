#! /usr/local/bin/octave -qf
pkg load image
IMG = imread('../output/6dilated_image.jpg');
FILL = imfill(IMG);
imwrite(FILL, '../output/7filled_image.png')
