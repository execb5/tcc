import cv2
import numpy as np
import settings


def invert_image(image):
    return cv2.bitwise_not(image)


def convert_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_bilateral_filter(image):
    return cv2.bilateralFilter(image, 9, 75, 75)


def apply_histogram_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def apply_morphological_opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def binarize_image(image):
    (thresh, binarized_image) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binarized_image


def apply_sobel_edge_detection(image):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def apply_dilation(image, element_size):
    kernel = np.ones((element_size, element_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


def apply_erosion(image, element_size):
    kernel = np.ones((element_size, element_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def find_contours(image):
    return cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]


def extract_region_of_interest(fill_dilated, original_image):
    contours = find_contours(fill_dilated)
    i = 10
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > w:
            continue
        cv2.boundingRect(contour)
        name = settings.output_path + str(i) + "roi.jpg"
        cv2.imwrite(name, original_image[y:y + h, x:x + w])
        i = i + 1
        rois.append(name)
    return rois


def imfill(image):
    contours = find_contours(image)
    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, 255, -1)
    return image


def resize_image(image):
    height = image.shape[0]
    width = image.shape[1]

    if width > height:
        biggest = width
        border_width = 0
        border_height = (width - height) / 2
    else:
        biggest = height
        border_height = 0
        border_width = (height - width) / 2

    border = cv2.copyMakeBorder(image,
                                top=border_height,
                                bottom=border_height,
                                left=border_width,
                                right=border_width,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[255, 255, 255])

    return cv2.resize(border, (biggest, biggest))
