import cv2
import numpy as np


def train_letter_model():
    samples = np.loadtxt('../learning_data/letters_samples.data', np.float32)
    responses = np.loadtxt('../learning_data/letters_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def train_number_model():
    samples = np.loadtxt('../learning_data/numbers_samples.data', np.float32)
    responses = np.loadtxt('../learning_data/numbers_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model
