import cv2
import numpy as np


class CharacterExtractor:
    def __init__(self):
        pass

    def extract_characters(self, image):
        clean, characters = self.extract_plate_characters(image)
        
        i = 9
        for _, char_img in characters:
            cv2.imwrite("../output/a" + str(i) + "character.png", char_img)
            i = i + 1
        return characters

    def extract_plate_characters(self, image):
        bw_image = cv2.bitwise_not(image)
        _, contours, _ = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        char_mask = np.zeros_like(image)
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
