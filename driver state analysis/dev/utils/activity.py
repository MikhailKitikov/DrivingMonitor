import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


def load_classes():

	classes = []
	
	with open("assets/classes.txt", 'r') as file:
		for line in file.readlines():
			classes.append(line.strip().split(': ')[1])

	return classes


def load_mobilenet_v2_distill():

    weights_path = 'models/mobilenet_v2_distill.hdf5'
    model = load_model(weights_path)

    return model
