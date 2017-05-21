import cv2

from plate_extractor import *
from plate_cleaner import *
from character_extractor import *
from character_reader import *


def main():
    image = cv2.imread("full_car.jpg")
    plate_extractor = PlateExtractor()
    plate_cleaner = PlateCleaner()
    character_extractor = CharacterExtractor()
    candidates = plate_extractor.extract_plate_candidates(image)
    for candidate in candidates:
        prepared_plate = plate_cleaner.clean_plate(candidate)
        characters = character_extractor.extract_characters(prepared_plate)

        letter_samples = np.loadtxt('../learning_data/letters_samples.data', np.float32)
        letter_responses = np.loadtxt('../learning_data/letters_responses.data', np.float32)
        letter_responses = letter_responses.reshape((letter_responses.size, 1))

        letter_model = cv2.ml.KNearest_create()
        letter_model.train(letter_samples, cv2.ml.ROW_SAMPLE, letter_responses)

        number_samples = np.loadtxt('../learning_data/numbers_samples.data', np.float32)
        number_responses = np.loadtxt('../learning_data/numbers_responses.data', np.float32)
        number_responses = number_responses.reshape((number_responses.size, 1))

        number_model = cv2.ml.KNearest_create()
        number_model.train(number_samples, cv2.ml.ROW_SAMPLE, number_responses)
        character_reader = CharacterReader(number_model, letter_model)

        print character_reader.read_characters(characters)


if __name__ == "__main__":
    main()
