import time
import cv2
from operations import *


class PlateCleaner:
    def __init__(self):
        pass

    def clean_plate(self, plate):
        start_time = time.time()
        gray_plate = convert_grayscale(plate)
        cv2.imwrite('../output/a01grayscale.jpg', gray_plate)
        end_time = time.time() - start_time
        print "grayscale " + str(end_time) + " seconds"

        start_time = time.time()
        fill_binary = binarize_image(gray_plate)
        cv2.imwrite('../output/a02fill_binary.jpg', fill_binary)
        end_time = time.time() - start_time
        print "binarization " + str(end_time) + " seconds"

        start_time = time.time()
        dilated_plate = apply_dilation(fill_binary, 11)
        cv2.imwrite('../output/a03dilated_image.jpg', dilated_plate)
        end_time = time.time() - start_time
        print "dilation " + str(end_time) + " seconds"

        start_time = time.time()
        fill_eroded = apply_erosion(dilated_plate, 20)
        cv2.imwrite('../output/a04fill_eroded.jpg', fill_eroded)
        end_time = time.time() - start_time
        print "erosion " + str(end_time) + " seconds"

        start_time = time.time()
        dilated_plate = apply_dilation(fill_eroded, 22)
        cv2.imwrite('../output/a05dilated_again_image.jpg', dilated_plate)
        end_time = time.time() - start_time
        print "dilation " + str(end_time) + " seconds"

        start_time = time.time()
        fill_eroded = apply_erosion(dilated_plate, 22)
        cv2.imwrite('../output/a06fill_eroded_again.jpg', fill_eroded)
        end_time = time.time() - start_time
        print "erosion " + str(end_time) + " seconds"
        return fill_eroded
