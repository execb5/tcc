import cv2
import numpy as np
import settings


def train_letter_model():
    samples = np.loadtxt(settings.learning_data_path + 'letters_samples.data', np.float32)
    responses = np.loadtxt(settings.learning_data_path + 'letters_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def train_number_model():
    samples = np.loadtxt(settings.learning_data_path + 'numbers_samples.data', np.float32)
    responses = np.loadtxt(settings.learning_data_path + 'numbers_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model
