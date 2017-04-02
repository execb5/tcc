import cv2
import numpy

def main():
    image = cv2.imread('test_image_2.png')
    gray_image = convert_grayscale(image)
    cv2.imwrite('grayscale.jpg', gray_image)

    equalized_image = apply_histogram_equalization(gray_image)
    cv2.imwrite('histogram_eq.jpg', equalized_image)

    # opened_image = apply_morphological_openning(equalized_image)
    # cv2.imwrite('opened_image.jpg', opened_image)

    binarized_image = binarize_image(equalized_image)
    cv2.imwrite('binarized_image.jpg', binarized_image)

    sobel_image = apply_sobel_edge_detection(binarized_image)
    cv2.imwrite('sobel_image.jpg', sobel_image)

    dilated_image = apply_dilation(sobel_image)
    cv2.imwrite('dilated_image.jpg', dilated_image)

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
    kernel = numpy.ones((5,5),numpy.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

if __name__ == "__main__":
    main()
