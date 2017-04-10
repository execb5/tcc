#! /usr/local/bin/octave -qf
pkg load image
IMG = imread('../img/6dilated_image.jpg');
FILL = imfill(IMG);
imwrite(FILL, '../img/7filled_image.png')
