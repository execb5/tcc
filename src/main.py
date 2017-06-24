import cv2
import sys
import imghdr
import settings

from multiprocessing import Process, Semaphore, Queue
from plate_extractor import *
from plate_cleaner import *
from character_extractor import *
from character_reader import *
from model_factory import *

number_model = 0
letter_model = 0
plate_extractor = 0
plate_cleaner = 0
character_extractor = 0
character_reader = 0


def main():
    settings.init()
    global number_model
    global letter_model
    global plate_extractor
    global plate_cleaner
    global character_extractor
    global character_reader
    number_model = train_number_model()
    letter_model = train_letter_model()
    plate_extractor = PlateExtractor()
    plate_cleaner = PlateCleaner()
    character_extractor = CharacterExtractor()
    character_reader = CharacterReader(number_model, letter_model)
    photo_counter = len(sys.argv) - 1
    q1 = Queue()
    Process(target=candidates_extractor_process, args=(q1, photo_counter)).start()
    for index, item in enumerate(sys.argv):
        if index == 0:
            continue
        if imghdr.what(item) in ['jpg', 'png', 'JPEG', 'jpeg', 'JPG', 'gif']:
            image = cv2.imread(item)
            bilateral = plate_extractor.get_bilateral(image)
            q1.put((image, bilateral))
        else:
            process_video(item)


def candidates_extractor_process(q1, photo_counter):
    q2 = Queue()
    Process(target=character_segmentator_process, args=(q2, photo_counter)).start()
    for i in range(photo_counter):
        image, bilateral = q1.get()
        filled_image = plate_extractor.get_imfill(bilateral)
        q2.put((image, filled_image))
        # new process:
        # fill_eroded = plate_extractor.erode_image(filled_image)
        # new process:
        # rois = plate_extractor.get_rois(fill_eroded, image)
        # for index, candidate in enumerate(rois):
            # prepared_plate = plate_cleaner.clean_plate(candidate, index)
            # characters = character_extractor.extract_characters(prepared_plate, index)
            # plate_read = character_reader.read_characters(characters)
            # if plate_read is not None:
                # print plate_read


def character_segmentator_process(q2, photo_counter):
    q3 = Queue()
    Process(target=character_reader_process, args=(q3, photo_counter)).start()
    for i in range(photo_counter):
        image, filled_image = q2.get()
        fill_eroded = plate_extractor.erode_image(filled_image)
        q3.put((image, fill_eroded))


def character_reader_process(q3, photo_counter):
    for i in range(photo_counter):
        image, fill_eroded = q3.get()
        rois = plate_extractor.get_rois(fill_eroded, image)
        for index, candidate in enumerate(rois):
            prepared_plate = plate_cleaner.clean_plate(candidate, index)
            characters = character_extractor.extract_characters(prepared_plate, index)
            plate_read = character_reader.read_characters(characters)
            if plate_read is not None:
                print plate_read


def process_frame_or_image(image, plate_from_file_name):
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

        plate_read = character_reader.read_characters(characters)


def process_video(item):
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


def get_license_plate_from_file_name(item):
    return item.split('.')[0].split('/')[-1]


if __name__ == "__main__":
    main()
