import time
import cv2
from operations import *


class PlateCleaner:
    def __init__(self):
        pass

    def clean_plate(self, plate):
        if __debug__:
            start_time = time.time()
        gray_plate = convert_grayscale(plate)
        if __debug__:
            cv2.imwrite('../output/a1grayscale.jpg', gray_plate)
            end_time = time.time() - start_time
            print "grayscale " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        fill_binary = binarize_image(gray_plate)
        if __debug__:
            cv2.imwrite('../output/a2fill_binary.jpg', fill_binary)
            end_time = time.time() - start_time
            print "binarization " + str(end_time) + " seconds"

        return fill_binary
