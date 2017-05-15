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
    character_reader = CharacterReader()
    plate = plate_extractor.extract_plate_candidates(image)
    prepared_plate = plate_cleaner.clean_plate(plate)
    characters = character_extractor.extract_characters(prepared_plate)
    character_reader.read_characters(characters)


if __name__ == "__main__":
    main()
