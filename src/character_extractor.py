import cv2
import settings
from operations import *


class CharacterExtractor:
    def __init__(self):
        pass

    def extract_characters(self, image, index):
        characters = self.extract_plate_characters(image)

        for (i, character_img) in enumerate(characters):
            cv2.imwrite(settings.output_path + "c" + str(index) + str(i) + "character.png", character_img)
        return characters

    def extract_plate_characters(self, image):
        bw_image = invert_image(image)
        filled_image = imfill(bw_image)

        if __debug__:
            cv2.imwrite(settings.output_path + 'b1filled_characters.png', filled_image)

        contours = find_contours(filled_image)
        char_mask = np.zeros_like(image)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            center = (x + w / 2, y + h / 2)
            if (area > 1000) and (area < 10000):
                x, y, w, h = x - 4, y - 4, w + 8, h + 8
                if x > 0 and y > 0:
                    bounding_boxes.append((center, (x, y, w, h)))
                    cv2.rectangle(char_mask, (x, y), (x + w, y + h), 255, -1)

        if __debug__:
            cv2.imwrite(settings.output_path + 'b2mask.png', char_mask)

        bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

        characters = []
        for _, bbox in bounding_boxes:
            x, y, w, h = bbox
            char_image = image[y:y + h, x:x + w]
            characters.append(char_image)

        return characters
