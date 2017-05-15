from PIL import Image
import pytesseract


class CharacterReader:
    letter_options = "-psm 10 -c tessedit_char_whitelist=QWERTYUIOPASDFGHJKLZXCVBNM"
    number_options = "-psm 10 -c tessedit_char_whitelist=1234567890"

    def __init__(self):
        pass

    def read_characters(self, characters):

        letters = characters[:3]
        numbers = characters[3:]

        for letter in letters:
            image = Image.fromarray(letter)
            print pytesseract.image_to_string(image, config=self.letter_options)

        for number in numbers:
            image = Image.fromarray(number)
            print pytesseract.image_to_string(image, config=self.number_options)
