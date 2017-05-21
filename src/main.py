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
    for candidate in candidates:
        prepared_plate = plate_cleaner.clean_plate(candidate)
        characters = character_extractor.extract_characters(prepared_plate)

        samples = np.loadtxt('char_samples.data', np.float32)
        responses = np.loadtxt('char_responses.data', np.float32)
        responses = responses.reshape((responses.size, 1))

        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, responses)

        i = 0
        plate_chars = ""
        for char_img in characters:
            resized = resize_image(char_img)
            small_img = cv2.resize(resized, (100, 100))
            cv2.imwrite('penis%s.png' % i, small_img)
            i = i + 1
            small_img = small_img.reshape((1, 10000))
            small_img = np.float32(small_img)
            retval, results, neigh_resp, dists = model.findNearest(small_img, k=3)
            plate_chars += str(chr((results[0][0])))

        print("Licence plate: %s" % plate_chars)


if __name__ == "__main__":
    main()
