from PIL import Image
import pytesseract


class CharacterReader:
    def __init__(self):
        pass

    def read_characters(self, characters):
        i = 0
        for _, character in characters:
            image = Image.fromarray(character)
            if i < 3:
                print pytesseract.image_to_string(image,
                                                  config="-psm 10 -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM")
            else:
                print pytesseract.image_to_string(image, config="-psm 10 -c tessedit_char_whitelist=1234567890")
            i = i + 1
