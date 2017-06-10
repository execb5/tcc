import cv2
import settings
from operations import *


class CharacterExtractor:
    def __init__(self):
        pass

    def extract_characters(self, image, index):
        characters = self.extract_plate_characters(image, index)

        for (i, character_img) in enumerate(characters):
            cv2.imwrite(settings.output_path + "c" + str(index) + str(i) + "character.png", character_img)
        return characters

    def extract_plate_characters(self, image, index):
        bw_image = invert_image(image)
        filled_image = imfill(bw_image)

        if __debug__:
            cv2.imwrite(settings.output_path + 'b1' + str(index) + 'filled_characters.png', filled_image)

        contours = find_contours(filled_image)
        char_mask = np.zeros_like(image)

        bounding_boxes = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width > height:
                continue
            area = width * height
            center = (x + width / 2, y + height / 2)
            if (area > 1000) and (area < 10000):
                x, y, width, height = x - 4, y - 4, width + 8, height + 8
                if x > 0 and y > 0:
                    bounding_boxes.append((center, (x, y, width, height)))
                    cv2.rectangle(char_mask, (x, y), (x + width, y + height), 255, -1)

        if __debug__:
            cv2.imwrite(settings.output_path + 'b2' + str(index) + 'mask.png', char_mask)

        bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

        characters = []
        for _, bbox in bounding_boxes:
            x, y, width, height = bbox
            char_image = image[y:y + height, x:x + width]
            characters.append(char_image)

        return characters
