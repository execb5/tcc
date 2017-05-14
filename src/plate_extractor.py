import time
import cv2
import numpy
from PIL import Image


class PlateExtractor:
    def __init__(self):
        pass

    def extract_plate_image(self, image):

        start_time = time.time()
        gray_image = self.convert_grayscale(image)
        cv2.imwrite('../output/01grayscale.jpg', gray_image)
        end_time = time.time() - start_time
        print "grayscale " + str(end_time) + " seconds"

        start_time = time.time()
        bilateral_image = self.apply_bilateral_filter(gray_image)
        cv2.imwrite('../output/02bilateral.jpg', bilateral_image)
        end_time = time.time() - start_time
        print "bilateral filter " + str(end_time) + " seconds"

        start_time = time.time()
        equalized_image = self.apply_histogram_equalization(bilateral_image)
        cv2.imwrite('../output/03histogram_eq.jpg', equalized_image)
        end_time = time.time() - start_time
        print "histogram equalization " + str(end_time) + " seconds"

        start_time = time.time()
        binarized_image = self.binarize_image(equalized_image)
        cv2.imwrite('../output/04binarized_image.jpg', binarized_image)
        end_time = time.time() - start_time
        print "binarize " + str(end_time) + " seconds"

        start_time = time.time()
        sobel_image = self.apply_sobel_edge_detection(binarized_image)
        cv2.imwrite('../output/05sobel_image.jpg', sobel_image)
        end_time = time.time() - start_time
        print "edge detection " + str(end_time) + " seconds"

        start_time = time.time()
        dilated_image = self.apply_dilation(sobel_image)
        cv2.imwrite('../output/06dilated_image.jpg', dilated_image)
        end_time = time.time() - start_time
        print "dilation " + str(end_time) + " seconds"

        start_time = time.time()
        filled_image = self.imfill(dilated_image)
        cv2.imwrite("../output/07filled_image.png", filled_image)
        end_time = time.time() - start_time
        print "imfill " + str(end_time) + " seconds"

        start_time = time.time()
        fill_eroded = self.apply_super_erosion(filled_image)
        cv2.imwrite('../output/10fill_eroded.jpg', fill_eroded)
        end_time = time.time() - start_time
        print "erosion " + str(end_time) + " seconds"

        start_time = time.time()
        fill_dilated = self.apply_super_dilation(fill_eroded)
        cv2.imwrite('../output/11fill_dilated.jpg', fill_dilated)
        end_time = time.time() - start_time
        print "super dilation " + str(end_time) + " seconds"

        start_time = time.time()
        rois = self.extract_region_of_interest(fill_dilated, image)
        end_time = time.time() - start_time
        print "rois " + str(end_time) + " seconds"
        return cv2.imread(rois[0])

    def convert_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_bilateral_filter(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    def apply_histogram_equalization(self, image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def apply_morphological_openning(self, image):
        kernel = numpy.ones((5, 5), numpy.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def binarize_image(self, image):
        (thresh, binarized_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binarized_image

    def apply_sobel_edge_detection(self, image):
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    def apply_dilation(self, image):
        kernel = numpy.ones((5, 5), numpy.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def apply_super_erosion(self, image):
        kernel = numpy.ones((110, 110), numpy.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def apply_super_dilation(self, image):
        kernel = numpy.ones((110, 110), numpy.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def extract_region_of_interest(self, fill_dilated, original_image):
        _, thresh = cv2.threshold(fill_dilated, 127, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        i = 12
        rois = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.boundingRect(contour)
            name = "../output/" + str(i) + "roi.jpg"
            cv2.imwrite(name, original_image[y:y + h, x:x + w])
            i = i + 1
            rois.append(name)
        return rois

    def imfill(self, gray):
        des = gray
        _, contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(des, [cnt], 0, 255, -1)
            gray = cv2.bitwise_not(des)
        return cv2.bitwise_not(gray)
