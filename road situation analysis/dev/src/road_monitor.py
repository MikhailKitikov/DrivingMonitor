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

from src.regulations.traffic_lights.traffic_light_service import TrafficLightService
from src.regulations.traffic_signs.traffic_sign_service import TrafficSignService
from src.regulations.violation.speed_limits_violation_service import SpeedLimitsViolationService

from src.road.lane_detection.lane_detection_service import LaneDetectionService
from src.road.state_estimation.state_estimation_service import StateEstimationService

from src.vehicles.accidents.accident_service import AccidentService
from src.vehicles.collisions.collision_service import CollisionService
from src.vehicles.detection_tracking.detection_tracking_service import DetectionTrackingService

 
class RoadMonitor:

	def __init__(self, args):

		# modules

		self.traffic_light_service = TrafficLightService()
		self.traffic_sign_service = TrafficSignService()
		self.speed_limits_violation_service = SpeedLimitsViolationService()
		self.lane_detection_service = LaneDetectionService()
		self.state_estimation_service = StateEstimationService()
		self.accident_service = AccidentService()
		self.collision_service = CollisionService()
		self.detection_tracking_service = DetectionTrackingService()
		
		# variables

		self.cnt = 0
		self.frame_queue = deque(maxlen=100)
		self.detection_queue = deque(maxlen=100)
		self.current_car_ids = set()
		self.prev_time = time.time()
		self.curr_max_speed = 110
		self.prev_ego_state = None

 
	def process(self, frame):

		new_time = time.time()
		dt = new_time - self.prev_time
		self.prev_time = new_time

		self.cnt += 1
		self.frame_queue.append(frame)

		# state estimation

		ego_state = self.state_estimation_service.process(frame, dt)
		print(f"[INFO] Current state: {ego_state}")

		# lane detection

		frame = self.lane_detection_service.process(frame)

		# traffic sign detection

		traffic_signs = self.traffic_sign_service.process(frame)
		for traffic_sign in traffic_signs:
			if traffic_sign.startswith('speed'):
				new_max_speed = int(traffic_sign[6:])
				print(f"Warning! New speed limit: {new_max_speed}")

		# traffic light detection

		traffic_light_color = self.traffic_light_service.process(frame)
		if traffic_light_color != 'green':
			print(f"Warning! Traffic light color: {traffic_light_color}")

		# dynamic object detection

		detection_result = self.detection_tracking_service.process(frame)
		detection_cars = detection_result['car']
		detection_pedestrians = detection_result['pedestrian']
		self.detection_queue.append(detection_cars)

		# tracking

		new_car_ids = set(car['id'] for car in detection_cars)
		self.current_car_ids = new_car_ids

		# collision prediction

		self.collision_service.process(self.detection_queue, current_car_ids, dt, curr_max_speed)

		# accident recognition

		self.accident_service.process(self.frame_queue, detection_result)

		# speed limit violation detection & number plate recognition

		self.speed_limits_violation_service.process(self.prev_ego_state, ego_state, self.frame_queue, self.detection_queue, 
			self.current_car_ids, new_time, dt, self.curr_max_speed)

		# update variables

		self.prev_ego_state = ego_state
 