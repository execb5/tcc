import cv2
import numpy as np

CHARS = [chr(ord('0') + i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]
# CHARS = ["2", "3", "4", "5", "6", "7", "8", "9", "0", "A", "B", "C", "D", "F", "G", "H", "J", "K", "L", "M",
#          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

print CHARS


def load_char_images():
    chars = {}
    for char in CHARS:
        char_img = cv2.imread("chars/%s.png" % char, 0)
        chars[char] = char_img
    return chars


characters = load_char_images()
samples = np.empty((0, 10000))
for char in CHARS:
    char_img = characters[char]
    small_char = cv2.resize(char_img, (100, 100))
    sample = small_char.reshape((1, 10000))
    cv2.imwrite('%s.png' % char, small_char)
    samples = np.append(samples, sample, 0)

responses = np.array([ord(c) for c in CHARS], np.float32)
responses = responses.reshape((responses.size, 1))

np.savetxt('char_samples.data', samples)
np.savetxt('char_responses.data', responses)
