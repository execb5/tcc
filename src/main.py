import cv2
import sys
import imghdr
import settings

from plate_extractor import *
from plate_cleaner import *
from character_extractor import *
from character_reader import *
from model_factory import *

photo_counter = 0
plates_read = 0
plates_read_correctly = 0


def main():
    settings.init()
    # photo_counter = len(sys.argv) - 1
    for index, item in enumerate(sys.argv):
        if index == 0:
            continue
        if __debug__:
            print imghdr.what(item)
        if imghdr.what(item) in ['jpg', 'png', 'JPEG', 'jpeg', 'JPG', 'gif']:
            image = cv2.imread(item)
            plate_from_file_name = get_license_plate_from_file_name(item)
            process_frame_or_image(image, plate_from_file_name)
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
    print 'Received %s image(s)' % photo_counter
    print 'Read %s license plates' % plates_read
    print 'Read %s license plates correctly' % plates_read_correctly
    print "%s%% of license plates were read" % ((plates_read * 100) / photo_counter)
    if plates_read > 0:
        print "%s%% of license plates were read correctly" % ((plates_read_correctly * 100) / plates_read)
    else:
        print "0% of license plates were read correctly"


def process_frame_or_image(image, plate_from_file_name):
    global photo_counter
    global plates_read
    global plates_read_correctly
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
        if plate_read is not None:
            is_correct = plate_read == plate_from_file_name
            print '%s == %s ? %s' % (plate_from_file_name, plate_read, is_correct)
            if is_correct:
                plates_read_correctly += 1
            plates_read += 1
            photo_counter += 1
            if not __debug__:
                return
    photo_counter += 1
    print "Didn't find anything in photo of plate %s :(" % plate_from_file_name
    return


def get_license_plate_from_file_name(item):
    return item.split('.')[0].split('/')[-1]


if __name__ == "__main__":
    main()
