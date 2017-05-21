import os
import cv2
from operations import *


def extract_chars(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]

    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x, y, w, h = x - 2, y - 2, w + 4, h + 4
        bounding_boxes.append((x, y, w, h))

    characters = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        char_image = img[y:y + h, x:x + w]
        characters.append(char_image)

    return characters


def output_chars(chars, labels):
    for i, char in enumerate(chars):
        filename = "../learning_data/chars/%s.png" % labels[i]
        char = resize_image(char)
        char = cv2.resize(char, (300, 300))
        cv2.imwrite(filename, char)


if not os.path.exists("../learning_data/chars"):
    os.makedirs("../learning_data/chars")

img_digits = cv2.imread("../learning_data/font/numbers.png", 0)
img_letters = cv2.imread("../learning_data/font/letters.png", 0)

digits = extract_chars(img_digits)
letters = extract_chars(img_letters)

DIGITS = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
LETTERS = [chr(ord('A') + i) for i in range(25, -1, -1)]

output_chars(digits, DIGITS)
output_chars(letters, LETTERS)
