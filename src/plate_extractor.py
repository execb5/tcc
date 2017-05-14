import time
import cv2
import numpy
from PIL import Image
from operations import *


class PlateExtractor:
    def __init__(self):
        pass

    def extract_plate_candidates(self, image):
        start_time = time.time()
        gray_image = convert_grayscale(image)
        cv2.imwrite('../output/01grayscale.jpg', gray_image)
        end_time = time.time() - start_time
        print "grayscale " + str(end_time) + " seconds"

        start_time = time.time()
        bilateral_image = apply_bilateral_filter(gray_image)
        cv2.imwrite('../output/02bilateral.jpg', bilateral_image)
        end_time = time.time() - start_time
        print "bilateral filter " + str(end_time) + " seconds"

        start_time = time.time()
        equalized_image = apply_histogram_equalization(bilateral_image)
        cv2.imwrite('../output/03histogram_eq.jpg', equalized_image)
        end_time = time.time() - start_time
        print "histogram equalization " + str(end_time) + " seconds"

        start_time = time.time()
        binarized_image = binarize_image(equalized_image)
        cv2.imwrite('../output/04binarized_image.jpg', binarized_image)
        end_time = time.time() - start_time
        print "binarize " + str(end_time) + " seconds"

        start_time = time.time()
        sobel_image = apply_sobel_edge_detection(binarized_image)
        cv2.imwrite('../output/05sobel_image.jpg', sobel_image)
        end_time = time.time() - start_time
        print "edge detection " + str(end_time) + " seconds"

        start_time = time.time()
        dilated_image = apply_dilation(sobel_image, 5)
        cv2.imwrite('../output/06dilated_image.jpg', dilated_image)
        end_time = time.time() - start_time
        print "dilation " + str(end_time) + " seconds"

        start_time = time.time()
        filled_image = imfill(dilated_image)
        cv2.imwrite("../output/07filled_image.png", filled_image)
        end_time = time.time() - start_time
        print "imfill " + str(end_time) + " seconds"

        start_time = time.time()
        fill_eroded = apply_erosion(filled_image, 110)
        cv2.imwrite('../output/10fill_eroded.jpg', fill_eroded)
        end_time = time.time() - start_time
        print "erosion " + str(end_time) + " seconds"

        start_time = time.time()
        fill_dilated = apply_dilation(fill_eroded, 110)
        cv2.imwrite('../output/11fill_dilated.jpg', fill_dilated)
        end_time = time.time() - start_time
        print "super dilation " + str(end_time) + " seconds"

        start_time = time.time()
        rois = extract_region_of_interest(fill_dilated, image)
        end_time = time.time() - start_time
        print "rois " + str(end_time) + " seconds"
        return cv2.imread(rois[0])
