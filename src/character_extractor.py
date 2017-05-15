import cv2
from operations import *


class CharacterExtractor:
    def __init__(self):
        pass

    def extract_characters(self, image):
        characters = self.extract_plate_characters(image)

        i = 9
        for character_img in characters:
            cv2.imwrite("../output/a" + str(i) + "character.png", character_img)
            i = i + 1
        return characters

    def extract_plate_characters(self, image):
        bw_image = invert_image(image)
        filled_image = imfill(bw_image)

        cv2.imwrite('../output/a7filled_characters.png', filled_image)

        contours = find_contours(filled_image)
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

        cv2.imwrite('../output/a8mask.png', char_mask)

        bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

        characters = []
        for center, bbox in bounding_boxes:
            x, y, w, h = bbox
            char_image = image[y:y + h, x:x + w]
            characters.append(char_image)

        return characters
