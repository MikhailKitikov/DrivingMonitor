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
from scipy.spatial import distance as dist
import argparse
import imutils
import time
import dlib
import cv2

 
class EyeBlinkService:

	def __init__(self, args):

		# build model

		self.face_model = args['face_model']

		print("[INFO] loading face detector...")
		if self.face_model == 'CNN':
			self.detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
		else:
			self.detector = dlib.get_frontal_face_detector()

		print("[INFO] loading landmark detector...")
		if self.face_model == 'CNN':
			self.predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
		else:
			self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
		
		# variables

		self.EYE_AR_THRESH = 0.3
		self.EYE_AR_CONSEC_FRAMES = 48

		self.ear_states = deque(maxlen=50)
		self.danger_scale_value_ear = 1

		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

 
	def process(self, args):

		frame = args['frame']
		ALARM_ON = args['ALARM_ON']
		danger_value_threshold = args['danger_value_threshold']

		ear = self.calculate_ear(frame)
		ear = ear if ear is not None else 0
		self.ear_states.append(int(ear < self.EYE_AR_THRESH))

		if len(self.ear_states) >= self.ear_states.maxlen:
			closed_eyes_frac = np.mean(self.ear_states)
			self.danger_scale_value_ear = max(1, min(10, int(round(closed_eyes_frac * 10))))
			ALARM_ON = (self.danger_scale_value_ear > danger_value_threshold) & (not ALARM_ON)

		return {
			'frame': frame,
			'ear': round(ear * 100, 2),
			'danger_scale_value': self.danger_scale_value_ear,
			'ALARM_ON': ALARM_ON,
			'sound_text': "Please! Watch the road!"
		}


	def calculate_ear(self, frame):

		# convert to gray
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces
		rects = self.detector(gray, 0)
		if not rects:
			return None
		rect = rects[0]

		# detect landmarks
		if self.face_model == 'CNN':
			rect = rect.rect

		# draw face boundaries
		pt1 = (rect.left(), rect.top())
		pt2 = (rect.right(), rect.bottom())
		cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 1)

		# get keypoints
		shape = self.predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# visualize keypoints
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (255, 255, 255), 2)
		
		# extract eye coordinates
		leftEye = shape[self.lStart: self.lEnd]
		rightEye = shape[self.rStart: self.rEnd]
		
		# compute average eye aspect ratio
		ear = (self.eye_aspect_ratio(leftEye) + self.eye_aspect_ratio(rightEye)) / 2
		
		# visualize eyes
		cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

		return ear


	def eye_aspect_ratio(self, eye):

		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])
		C = dist.euclidean(eye[0], eye[3])

		return (A + B) / (2.0 * C)


	def rotate_image(self, image, angle):

		image_center = tuple(np.array(image.shape[:2]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_CUBIC)

		return result
 