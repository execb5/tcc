import time
import cv2
import numpy
from PIL import Image
import pytesseract

from operations import *
from plate_extractor import *
from plate_cleaner import *


def main():
    image = cv2.imread("full_car.jpg")
    plate_extractor = PlateExtractor()
    plate_cleaner = PlateCleaner()
    plate = plate_extractor.extract_plate_candidates(image)
    prepared_plate = plate_cleaner.clean_plate(plate)
    characters = extract_characters(prepared_plate)
    read_characters(characters)


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
