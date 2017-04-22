import cv2
import numpy
from PIL import Image
import pytesseract
from scipy import ndimage
from subprocess import call

def main():
    image = cv2.imread("full_car.jpg")

    gray_image = convert_grayscale(image)
    cv2.imwrite('../output/1grayscale.jpg', gray_image)

    bilateral_image = apply_bilateral_filter(gray_image)
    cv2.imwrite('../output/2bilateral.jpg', bilateral_image)

    equalized_image = apply_histogram_equalization(bilateral_image)
    cv2.imwrite('../output/3histogram_eq.jpg', equalized_image)

    binarized_image = binarize_image(equalized_image)
    cv2.imwrite('../output/4binarized_image.jpg', binarized_image)

    sobel_image = apply_sobel_edge_detection(binarized_image)
    cv2.imwrite('../output/5sobel_image.jpg', sobel_image)

    dilated_image = apply_dilation(sobel_image)
    cv2.imwrite('../output/6dilated_image.jpg', dilated_image)

    call(["./octave_imfill.m"])

    filled_image = cv2.imread("../output/7filled_image.png")

    fill_grayscale = convert_grayscale(filled_image)
    cv2.imwrite('../output/8fill_grayscale.jpg', fill_grayscale)

    fill_binary = binarize_image(fill_grayscale)
    cv2.imwrite('../output/9fill_binary.jpg', fill_binary)

    fill_eroded = apply_super_erosion(fill_binary)
    cv2.imwrite('../output/10fill_eroded.jpg', fill_eroded)

    fill_dilated = apply_super_dilation(fill_eroded)
    cv2.imwrite('../output/11fill_dilated.jpg', fill_dilated)

    rois = extract_region_of_interest(fill_dilated, image)
    print rois

def extract_region_of_interest(fill_dilated, original_image):
    _, thresh = cv2.threshold(fill_dilated, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 1
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.boundingRect(contour)
        name = "../output/roi_" + str(i) + ".jpg"
        cv2.imwrite(name, original_image[y:y+h,x:x+w])
        i=i+1
        rois.append(name)
    return rois

def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)

def apply_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_morphological_openning(image):
    kernel = numpy.ones((5,5),numpy.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def binarize_image(image):
    (thresh, binarized_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binarized_image

def apply_sobel_edge_detection(image):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize = 3, scale = scale, delta = delta,borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)

def apply_dilation(image):
    kernel = numpy.ones((5, 5),numpy.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def apply_super_erosion(image):
    kernel = numpy.ones((110, 110),numpy.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def apply_super_dilation(image):
    kernel = numpy.ones((110, 110),numpy.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

if __name__ == "__main__":
    main()
