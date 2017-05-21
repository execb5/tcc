import cv2
import numpy as np

NUMBERS = [chr(ord('0') + i) for i in range(10)]
LETTERS = [chr(ord('A') + i) for i in range(26)]


def load_char_images(array):
    chars = {}
    for char in array:
        char_img = cv2.imread(settings.learning_data_chars_path + "%s.png" % char, 0)
        chars[char] = char_img
    return chars


def create_samples(array, name):
    characters = load_char_images(array)
    samples = np.empty((0, 10000))
    for char in array:
        char_img = characters[char]
        small_char = cv2.resize(char_img, (100, 100))
        sample = small_char.reshape((1, 10000))
        samples = np.append(samples, sample, 0)

    responses = np.array([ord(c) for c in array], np.float32)
    responses = responses.reshape((responses.size, 1))

    np.savetxt(settings.learning_data_path + name + '_samples.data', samples)
    np.savetxt(settings.learning_data_path + name + '_responses.data', responses)


create_samples(NUMBERS, "numbers")
create_samples(LETTERS, "letters")
