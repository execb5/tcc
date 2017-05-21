import cv2

from plate_extractor import *
from plate_cleaner import *
from character_extractor import *


def main():
    image = cv2.imread("full_car.jpg")
    plate_extractor = PlateExtractor()
    plate_cleaner = PlateCleaner()
    character_extractor = CharacterExtractor()
    candidates = plate_extractor.extract_plate_candidates(image)
    for candidate in reversed(candidates):
        prepared_plate = plate_cleaner.clean_plate(candidate)
        characters = character_extractor.extract_characters(prepared_plate)

        letter_samples = np.loadtxt('letters_samples.data', np.float32)
        letter_responses = np.loadtxt('letters_responses.data', np.float32)
        letter_responses = letter_responses.reshape((letter_responses.size, 1))

        letter_model = cv2.ml.KNearest_create()
        letter_model.train(letter_samples, cv2.ml.ROW_SAMPLE, letter_responses)

        number_samples = np.loadtxt('numbers_samples.data', np.float32)
        number_responses = np.loadtxt('numbers_responses.data', np.float32)
        number_responses = number_responses.reshape((number_responses.size, 1))

        number_model = cv2.ml.KNearest_create()
        number_model.train(number_samples, cv2.ml.ROW_SAMPLE, number_responses)

        i = 0
        plate_chars = ""
        for char_img in characters:
            resized = resize_image(char_img)
            small_img = cv2.resize(resized, (100, 100))
            cv2.imwrite('penis%s.png' % i, small_img)
            small_img = small_img.reshape((1, 10000))
            small_img = np.float32(small_img)
            if i < 3:
                retval, results, neigh_resp, dists = letter_model.findNearest(small_img, k=1)
            else:
                retval, results, neigh_resp, dists = number_model.findNearest(small_img, k=1)
            plate_chars += str(chr((results[0][0])))
            i = i + 1

        print("Licence plate: %s" % plate_chars)


if __name__ == "__main__":
    main()
