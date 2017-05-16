import cv2
import sys

from plate_extractor import *
from plate_cleaner import *
from character_extractor import *
from character_reader import *

VIDEO = True

plate_extractor = PlateExtractor()
plate_cleaner = PlateCleaner()
character_extractor = CharacterExtractor()
character_reader = CharacterReader()


def main():
    if VIDEO:
        cap = cv2.VideoCapture("./video.mp4")
        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while True:
            _, frame = cap.read()
            try:
                aux(frame)
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
    else:
        image = cv2.imread("full_car.jpg")
        aux(image)

def aux(image):
    candidates = plate_extractor.extract_plate_candidates(image)
    for candidate in candidates:
        prepared_plate = plate_cleaner.clean_plate(candidate)
        characters = character_extractor.extract_characters(prepared_plate)
        character_reader.read_characters(characters)


if __name__ == "__main__":
    main()
