import time
import cv2
import numpy
from PIL import Image
import pytesseract

from operations import *
from plate_extractor import PlateExtractor


def main():
    image = cv2.imread("full_car.jpg")
    plate_extractor = PlateExtractor()
    plate = plate_extractor.extract_plate_candidates(image)
    prepared_plate = prepare_plate_image_for_tesseract(plate)
    characters = extract_characters(prepared_plate)
    read_characters(characters)


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
            print pytesseract.image_to_string(image,
                                              config="-psm 10 -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM")
        else:
            print pytesseract.image_to_string(image, config="-psm 10 -c tessedit_char_whitelist=1234567890")
        i = i + 1


def extract_plate_characters(image):
    bw_image = cv2.bitwise_not(image)
    _, contours, _ = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    char_mask = numpy.zeros_like(image)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w / 2, y + h / 2)
        if (area > 1000) and (area < 10000):
            x, y, w, h = x - 4, y - 4, w + 8, h + 8
            bounding_boxes.append((center, (x, y, w, h)))
            cv2.rectangle(char_mask, (x, y), (x + w, y + h), 255, -1)

    cv2.imwrite('../output/a07mask.png', char_mask)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask=bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    characters = []
    for center, bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = clean[y:y + h, x:x + w]
        characters.append((bbox, char_image))

    return clean, characters


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x, y, w, h = bbox
        cv2.rectangle(output_img, (x, y), (x + w, y + h), 255, 1)

    return output_img


if __name__ == "__main__":
    main()
