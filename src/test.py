import cv2
import numpy
from PIL import Image
import pytesseract
from scipy import ndimage

def after_matlab():
    image = cv2.imread("imgfill.png")

    fill_grayscale = convert_grayscale(image)
    cv2.imwrite('fill_grayscale.jpg', fill_grayscale)

    fill_binary = binarize_image(fill_grayscale)
    cv2.imwrite('fill_binary.jpg', fill_binary)

    fill_eroded = apply_erosion(fill_binary)
    cv2.imwrite('fill_eroded.jpg', fill_eroded)

    fill_dilated = apply_dilation(fill_eroded)
    cv2.imwrite('fill_dilated.jpg', fill_dilated)

    # fill_open = apply_morphological_openning(fill_binary)
    # cv2.imwrite("fill_open.jpg", fill_open)


def tesseract():
    letters = cv2.imread("letras.png")
    numbers = cv2.imread("numeros.png")

    print(pytesseract.image_to_string(Image.fromarray(letters), None, False, "-c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM -psm 6"))
    print(pytesseract.image_to_string(Image.fromarray(numbers), None, False, "-c tessedit_char_whitelist=0123456789 -psm 6"))

def main():
    image = cv2.imread("full_car.jpg")

    gray_image = convert_grayscale(image)
    cv2.imwrite('grayscale.jpg', gray_image)

    bilateral_image = apply_bilateral_filter(gray_image)
    cv2.imwrite('bilateral.jpg', bilateral_image)

    equalized_image = apply_histogram_equalization(bilateral_image)
    cv2.imwrite('histogram_eq.jpg', equalized_image)

    binarized_image = binarize_image(equalized_image)
    cv2.imwrite('binarized_image.jpg', binarized_image)

    sobel_image = apply_sobel_edge_detection(binarized_image)
    cv2.imwrite('sobel_image.jpg', sobel_image)

    dilated_image = apply_dilation(sobel_image)
    cv2.imwrite('dilated_image.jpg', dilated_image)

def test():
    test_grayscale()
    test_bilateral_filter()
    test_histogram_equalization()
    test_opening()
    test_subtraction()
    test_binarization()
    test_edge_detection()
    test_dilation()

def test_grayscale():
    image = cv2.imread('../img/original_image_1.png')
    gray_image = convert_grayscale(image)
    cv2.imwrite('grayscale.jpg', gray_image)

def test_bilateral_filter():
    image = cv2.imread('../img/original_image_2_gray_scale.png')

def test_histogram_equalization():
    image = cv2.imread('../img/original_image_3_bilateral_filter.png')
    grayscale = convert_grayscale(image)
    equalized_image = apply_histogram_equalization(grayscale)
    cv2.imwrite('histogram_eq.jpg', equalized_image)

def test_opening():
    image = cv2.imread('../img/original_image_4_histogram_equalization.png')
    opened_image = apply_morphological_openning(image)
    cv2.imwrite('opened_image.jpg', opened_image)

def test_subtraction():
    image = cv2.imread('../img/original_image_5_opening.png')

def test_binarization():
    image = cv2.imread('../img/original_image_6_subtraction.png')
    grayscale = convert_grayscale(image)
    binarized_image = binarize_image(grayscale)
    cv2.imwrite('binarized_image.jpg', binarized_image)

def test_edge_detection():
    image = cv2.imread('../img/original_image_7_binarization.png')
    sobel_image = apply_sobel_edge_detection(image)
    cv2.imwrite('sobel_image.jpg', sobel_image)

def test_dilation():
    image = cv2.imread('../img/original_image_8_edge_detection.png')
    dilated_image = apply_dilation(image)
    cv2.imwrite('dilated_image.jpg', dilated_image)

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
    kernel = numpy.ones((100, 100),numpy.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def apply_erosion(image):
    kernel = numpy.ones((100, 100),numpy.uint8)
    return cv2.erode(image, kernel, iterations = 1)

if __name__ == "__main__":
    # test()
    # main()
    # tesseract()
    after_matlab()
