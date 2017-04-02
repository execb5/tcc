import cv2
import numpy

def main():
    image = cv2.imread('image.jpg')
    gray_image = convert_grayscale(image)
    cv2.imwrite('grayscale.jpg', gray_image)
    equalized_image = apply_histogram_equalization(gray_image)
    cv2.imwrite('histogram_eq.jpg', equalized_image)
    # opened_image = apply_morphological_openning(equalized_image)
    # cv2.imwrite('opened_image.jpg', opened_image)
    binarized_image = binarize_image(equalized_image)
    cv2.imwrite('binarized_image.jpg', binarized_image)

def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def apply_morphological_openning(image):
    kernel = numpy.ones((5,5),numpy.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def binarize_image(image):
    (thresh, binarized_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binarized_image

if __name__ == "__main__":
    main()
