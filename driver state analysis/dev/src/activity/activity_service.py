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
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque
from tensorflow.keras.models import load_model

 
class ActivityService:

	def __init__(self, args):

		# build model

		self.activity_model_ = args['activity_model_']

		print("[INFO] loading activity model...")
		if self.activity_model_ == 'mobilenet-v2-distill':
			self.activity_model = self.load_mobilenet_v2_distill()
		else:
			raise Exception("Unknown model: {}".format(activity_model))
		
		# variables

		self.classes = self.load_classes()
		self.activity_states = deque(maxlen=10)
		self.activity_states.append(0)
		self.danger_scale_value_activity = 1

 
	def process(self, args):

		frame = args['frame']
		ALARM_ON = args['ALARM_ON']
		danger_value_threshold = args['danger_value_threshold']

		activity_class = self.predict_activity(frame)
		self.activity_states.append(int(activity_class != 0))

		if len(self.activity_states) >= self.activity_states.maxlen:
			bad_activity_frac = np.mean(self.activity_states)
			self.danger_scale_value_activity = max(1, min(10, int(round(bad_activity_frac * 10))))
			ALARM_ON = (self.danger_scale_value_activity > danger_value_threshold) & (not ALARM_ON)

		return {
			'frame': frame,
			'activity_class': self.get_last_activity_class(),
			'danger_scale_value': self.danger_scale_value_activity,
			'ALARM_ON': ALARM_ON,
			'sound_text': "Please! Watch the road!"
		}


	def predict_activity(self, frame):

		frame = cv2.resize(frame, (224, 224)).astype('float32')
		if self.activity_model_.startswith('mobilenet-v2'):
			frame = preprocess_input(frame)

		pred = self.activity_model(frame[np.newaxis, ...])[0]
		pred = np.argmax(pred)

		return pred


	def load_classes(self):

		classes = []
		
		with open("assets/classes.txt", 'r') as file:
			for line in file.readlines():
				classes.append(line.strip().split(': ')[1])

		return classes


	def load_mobilenet_v2_distill(self):

		weights_path = 'models/mobilenet_v2_distill.hdf5'
		model = load_model(weights_path)

		return model


	def get_last_activity_class(self):
		return self.classes[self.activity_states[-1]]
 