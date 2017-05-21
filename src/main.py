import cv2

from plate_extractor import *
from plate_cleaner import *
from character_extractor import *
from character_reader import *
from model_factory import *


def main():
    image = cv2.imread("full_car.jpg")
    number_model = train_number_model()
    letter_model = train_letter_model()
    plate_extractor = PlateExtractor()
    plate_cleaner = PlateCleaner()
    character_extractor = CharacterExtractor()
    candidates = plate_extractor.extract_plate_candidates(image)
    for candidate in candidates:
        prepared_plate = plate_cleaner.clean_plate(candidate)
        characters = character_extractor.extract_characters(prepared_plate)
        character_reader = CharacterReader(number_model, letter_model)
        print character_reader.read_characters(characters)


if __name__ == "__main__":
    main()
