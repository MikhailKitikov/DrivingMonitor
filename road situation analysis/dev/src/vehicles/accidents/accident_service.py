import tkinter
from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import time
import argparse
import os
from keras import backend as K
import tensorflow as tf
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pickle
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque
from tensorflow.keras.models import load_model

 
class AccidentService:

	def __init__(self, args):

		# build models

		print("[INFO] loading accident recognition model...")
		self.classifier = load_model('models/accident_classifier.hdf5')

		# variables

		self.were_in_accident = set()
		self.label_mapping = {
			0: 'normal',
			1: 'accident'
		}
		self.num_accidents = 0

 
	def process(self, frame_queue, detection_result):

		for det in detection_result:
			x_min, y_min, x_max, y_max = det['2d_bbox']
			x_min = max(0, x_min - 50)
			y_min = max(0, y_min - 50)
			x_max = min(frame_queue[-1].shape[1] - 1, x_max + 50)
			y_max = min(frame_queue[-1].shape[0] - 1, y_max + 50)

			crop = frame_queue[-1][x_min: x_max + 1, y_min: y_max + 1, :]
			crop = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)
			pred = self.classifier.predict(np.expand_dims(crop, 0))[0]
			is_accident = self.label_mapping[pred]

			if not det['id'] in self.were_in_accident:
				print("Warning! Accident detected.")
				self.num_accidents += 1
				self.were_in_accident.add(det['id'])

				pickle_data = {'type': "accident", 'frames': frame_queue}
				pickle_path = f"database/accidents/{self.num_accidents}.pkl"
				pickle.dump(pickle_data, open(pickle_path, 'wb'))
 