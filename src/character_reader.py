import cv2
from operations import *


class CharacterReader:
    def __init__(self, number_model, letter_model):
        self.number_model = number_model
        self.letter_model = letter_model

    def read_characters(self, characters):
        letters = characters[:3]
        numbers = characters[3:]
        chars = ""
        for char_img in letters:
            chars += self.read_character(char_img, self.letter_model)
        chars += "-"
        for char_img in numbers:
            chars += self.read_character(char_img, self.number_model)

        return chars

    def read_character(self, character, model):
        resized = resize_image(character)
        small_img = cv2.resize(resized, (100, 100))
        small_img = small_img.reshape((1, 10000))
        small_img = np.float32(small_img)

        retval, results, neigh_resp, dists = model.findNearest(small_img, k=1)

        return str(chr((results[0][0])))
