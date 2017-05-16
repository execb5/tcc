import time
import cv2
from operations import *


class PlateExtractor:
    def __init__(self):
        pass

    def extract_plate_candidates(self, image):
        if __debug__:
            start_time = time.time()
        gray_image = convert_grayscale(image)
        if __debug__:
            cv2.imwrite('../output/1grayscale.jpg', gray_image)
            end_time = time.time() - start_time
            print "grayscale " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        bilateral_image = apply_bilateral_filter(gray_image)
        if __debug__:
            cv2.imwrite('../output/2bilateral.jpg', bilateral_image)
            end_time = time.time() - start_time
            print "bilateral filter " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        equalized_image = apply_histogram_equalization(bilateral_image)
        if __debug__:
            cv2.imwrite('../output/3histogram_eq.jpg', equalized_image)
            end_time = time.time() - start_time
            print "histogram equalization " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        binarized_image = binarize_image(equalized_image)
        if __debug__:
            cv2.imwrite('../output/4binarized_image.jpg', binarized_image)
            end_time = time.time() - start_time
            print "binarize " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        sobel_image = apply_sobel_edge_detection(binarized_image)
        if __debug__:
            cv2.imwrite('../output/5sobel_image.jpg', sobel_image)
            end_time = time.time() - start_time
            print "edge detection " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        dilated_image = apply_dilation(sobel_image, 5)
        if __debug__:
            cv2.imwrite('../output/6dilated_image.jpg', dilated_image)
            end_time = time.time() - start_time
            print "dilation " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        filled_image = imfill(dilated_image)
        if __debug__:
            cv2.imwrite("../output/7filled_image.png", filled_image)
            end_time = time.time() - start_time
            print "imfill " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        fill_eroded = apply_erosion(filled_image, 110)
        if __debug__:
            cv2.imwrite('../output/8fill_eroded.jpg', fill_eroded)
            end_time = time.time() - start_time
            print "erosion " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        fill_dilated = apply_dilation(fill_eroded, 110)
        if __debug__:
            cv2.imwrite('../output/9fill_dilated.jpg', fill_dilated)
            end_time = time.time() - start_time
            print "super dilation " + str(end_time) + " seconds"

        if __debug__:
            start_time = time.time()
        names = extract_region_of_interest(fill_dilated, image)
        if __debug__:
            end_time = time.time() - start_time
            print "rois " + str(end_time) + " seconds"
        rois = []
        for name in names:
            rois.append(cv2.imread(name))
        return rois
