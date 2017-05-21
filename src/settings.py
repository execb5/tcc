import os

def init():
    dir = os.path.dirname(__file__)
    global output_path
    output_path = os.path.join(dir, '../output/')
    global learning_data_path
    learning_data_path = os.path.join(dir, '../learning_data/')
    global learning_data_chars_path
    learning_data_chars_path = os.path.join(dir, '../learning_data/chars/')
    global learning_data_font_path
    learning_data_font_path = os.path.join(dir, '../learning_data/font/')
