import time
import cv2
import numpy
from PIL import Image
import pytesseract
from scipy import ndimage
from subprocess import call

def main():
    plate = extract_plate_image()
    prepared_plate = prepare_plate_image_for_tesseract(plate)
    characters = extract_characters(prepared_plate)
    read_characters(characters)

def extract_plate_image():
    image = cv2.imread("full_car.jpg")

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
    dilated_image = apply_dilation(sobel_image)
    cv2.imwrite('../output/06dilated_image.jpg', dilated_image)
    end_time = time.time() - start_time
    print "dilation " + str(end_time) + " seconds"

    start_time = time.time()
    filled_image = imfill(dilated_image)
    cv2.imwrite("../output/07filled_image.png", filled_image)
    end_time = time.time() - start_time
    print "imfill " + str(end_time) + " seconds"

    start_time = time.time()
    fill_eroded = apply_super_erosion(filled_image)
    cv2.imwrite('../output/10fill_eroded.jpg', fill_eroded)
    end_time = time.time() - start_time
    print "erosion " + str(end_time) + " seconds"

    start_time = time.time()
    fill_dilated = apply_super_dilation(fill_eroded)
    cv2.imwrite('../output/11fill_dilated.jpg', fill_dilated)
    end_time = time.time() - start_time
    print "super dilation " + str(end_time) + " seconds"

    start_time = time.time()
    rois = extract_region_of_interest(fill_dilated, image)
    end_time = time.time() - start_time
    print "rois " + str(end_time) + " seconds"
    return cv2.imread(rois[0])

def prepare_plate_image_for_tesseract(plate):
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
    kernel = numpy.ones((11, 11),numpy.uint8)
    dilated_plate = cv2.dilate(fill_binary, kernel, iterations = 1)
    cv2.imwrite('../output/a03dilated_image.jpg', dilated_plate)
    end_time = time.time() - start_time
    print "dilation " + str(end_time) + " seconds"

    start_time = time.time()
    kernel = numpy.ones((20, 20),numpy.uint8)
    fill_eroded = cv2.erode(dilated_plate, kernel, iterations = 1)
    cv2.imwrite('../output/a04fill_eroded.jpg', fill_eroded)
    end_time = time.time() - start_time
    print "erosion " + str(end_time) + " seconds"

    start_time = time.time()
    kernel = numpy.ones((22, 22),numpy.uint8)
    dilated_plate = cv2.dilate(fill_eroded, kernel, iterations = 1)
    cv2.imwrite('../output/a05dilated_again_image.jpg', dilated_plate)
    end_time = time.time() - start_time
    print "dilation " + str(end_time) + " seconds"

    start_time = time.time()
    kernel = numpy.ones((22, 22),numpy.uint8)
    fill_eroded = cv2.erode(dilated_plate, kernel, iterations = 1)
    cv2.imwrite('../output/a06fill_eroded_again.jpg', fill_eroded)
    end_time = time.time() - start_time
    print "erosion " + str(end_time) + " seconds"
    return fill_eroded

def extract_characters(image):
    clean, characters = extract_plate_characters(image)
    output_img = highlight_characters(clean, characters)
    cv2.imwrite('../output/a08character_borders.png', output_img)
    i = 9
    for _, char_img in characters:
        cv2.imwrite("../output/a" + str(i) + "character.png", char_img)
        i = i + 1
    return characters

def read_characters(characters):
    i = 0
    for _, character in characters:
        image = Image.fromarray(character)
        if i < 3:
            print pytesseract.image_to_string(image, config="-psm 10 -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM")
        else:
            print pytesseract.image_to_string(image, config="-psm 10 -c tessedit_char_whitelist=1234567890")
        i = i + 1

def imfill(gray):
    des = gray
    _, contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(des,[cnt],0,255,-1)
        gray = cv2.bitwise_not(des)
    return cv2.bitwise_not(gray)

def extract_plate_characters(image):
    bw_image = cv2.bitwise_not(image)
    _, contours, _ = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    char_mask = numpy.zeros_like(image)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        if (area > 1000) and (area < 10000):
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)

    cv2.imwrite('../output/a07mask.png', char_mask)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        characters.append((bbox, char_image))

    return clean, characters

def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x,y,w,h = bbox
        cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)

    return output_img

def extract_region_of_interest(fill_dilated, original_image):
    _, thresh = cv2.threshold(fill_dilated, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    i = 12
    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.boundingRect(contour)
        name = "../output/" + str(i) + "roi.jpg"
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
