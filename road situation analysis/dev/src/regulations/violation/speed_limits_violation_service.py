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

from src.vehicles.number_plate_recognition.number_plate_recognition_service import NumberPlateRecognitionService 

 
class SpeedLimitsViolationService:

	def __init__(self, args):

		# modules

		self.number_plate_recognition_service = NumberPlateRecognitionService()
		
		# variables

		self.violators_queue = deque(maxlen=3)
		self.last_time_violated = dict()
		self.num_violations = 0


	def find_by_id(self, detection_queue, car_id):

		for obj in detection_queue:
			if obj['id'] == car_id:
				return obs

		return None

 
	def process(self, prev_ego_state, ego_state, frame_queue, detection_queue, current_car_ids, curr_time, dt, max_speed):

		new_violation_candidates = set()
		result_violators = set()

		# find violators

		for car_id in current_car_ids:

			if len(detection_queue) < 2 or self.find_by_id(detection_queue[-2], car_id) is None:
				continue

			prev_car_state = self.find_by_id(detection_queue[-2], car_id)
			curr_car_state = self.find_by_id(detection_queue[-1], car_id)

			prev_car_x = prev_car_state['center'][0] + prev_ego_state['position_x']
			prev_car_y = prev_car_state['center'][1] + prev_ego_state['position_y']

			curr_car_x = curr_car_state['center'][0] + ego_state['position_x']
			curr_car_y = curr_car_state['center'][1] + ego_state['position_y']

			pos_shift_x, pos_shift_y = abs(curr_car_x - prev_car_x), abs(curr_car_y - prev_car_y)
			pos_shift = np.sqrt(pos_shift_x ** 2 + pos_shift_y ** 2)
			speed = pos_shift / dt

			if speed >= max_speed + 20:
				new_violation_candidates.add(car_id)
				self.num_violations += 1
				print("Warning! Speed limit violation detected.")

				if (len(self.violators_queue) >= 2) and (car_id in self.violators_queue[-1]) and \
					(car_id in self.violators_queue[-2]) and (car_id not in self.last_time_violated or \
						curr_time - self.last_time_violated[car_id] > 60):

					result_violators.append(car_id)
					self.last_time_violated[car_id] = curr_time

		self.violators_queue.append(new_violation_candidates)

		# recognize number plates

		for car_id in result_violators:

			det = self.find_by_id(detection_queue[-1], car_id)
			x_min, y_min, x_max, y_max = det['2d_bbox']
			x_min = max(0, x_min - 50)
			y_min = max(0, y_min - 50)
			x_max = min(frame_queue[-1].shape[1] - 1, x_max + 50)
			y_max = min(frame_queue[-1].shape[0] - 1, y_max + 50)

			crop = frame_queue[-1][x_min: x_max + 1, y_min: y_max + 1, :]
			number_plate = self.number_plate_recognition_service.process(crop)['number_plate']

			if number_plate is not None:
				pickle_data = {'type': "speed limit violation", 'number plate': number_plate, 'frames': frame_queue}
				pickle_path = f"database/speed_limit_violations/{self.num_violations}.pkl"
				pickle.dump(pickle_data, open(pickle_path, 'wb'))
