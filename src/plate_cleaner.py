import time
import cv2
from operations import *


class PlateCleaner:
    def __init__(self):
        pass

    def clean_plate(self, plate, index):
        if __debug__:
            start_time = time.time()
        gray_plate = convert_grayscale(plate)
        if __debug__:
            cv2.imwrite('../output/a%d1grayscale.jpg' % index, gray_plate)
            end_time = time.time() - start_time
            print "grayscale " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        fill_binary = binarize_image(gray_plate)
        if __debug__:
            cv2.imwrite('../output/a%d2fill_binary.jpg' % index, fill_binary)
            end_time = time.time() - start_time
            print "binarization " + str(end_time) + " seconds"

        return fill_binary
