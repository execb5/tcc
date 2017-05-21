import cv2
import sys
import imghdr

from plate_extractor import *
from plate_cleaner import *
from character_extractor import *
from character_reader import *
from model_factory import *


def main():
    for index, item in enumerate(sys.argv):
        if index == 0:
            continue
        print imghdr.what(item)
        if imghdr.what(item) in ['jpg', 'png', 'JPEG', 'jpeg', 'JPG', 'gif']:
            image = cv2.imread(item)
            process_frame_or_image(image)
        else:
            cap = cv2.VideoCapture(item)
            while True:
                _, frame = cap.read()
                try:
                    process_frame_or_image(frame)
                except IOError as (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except ValueError:
                    print "Could not convert data to an integer."
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise

                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if __debug__:
                    print str(pos_frame) + " frames"
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    break


def process_frame_or_image(image):
    number_model = train_number_model()
    letter_model = train_letter_model()
    plate_extractor = PlateExtractor()
    plate_cleaner = PlateCleaner()
    character_extractor = CharacterExtractor()
    candidates = plate_extractor.extract_plate_candidates(image)
    for index, candidate in enumerate(candidates):
        prepared_plate = plate_cleaner.clean_plate(candidate, index)
        characters = character_extractor.extract_characters(prepared_plate, index)
        character_reader = CharacterReader(number_model, letter_model)
        print character_reader.read_characters(characters)


if __name__ == "__main__":
    main()
