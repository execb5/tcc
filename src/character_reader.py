from PIL import Image
import pytesseract


class CharacterReader:
    letter_options = "-psm 10 -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM"
    number_options = "-psm 10 -c tessedit_char_whitelist=1234567890"

    def __init__(self):
        pass

    def read_characters(self, characters):
        i = 0

        for _, character in characters:
            image = Image.fromarray(character)
            if i < 3:
                print pytesseract.image_to_string(image, config=self.letter_options)
            else:
                print pytesseract.image_to_string(image, config=self.number_options)
            i = i + 1
