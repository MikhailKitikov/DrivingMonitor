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

from src.activity_service import ActivityService
from src.eye_blink_service import EyeBlinkService

 
class DriverMonitor:

	def __init__(self, args):

		# modules

		self.activity_service = ActivityService({'activity_model_': args['activity_model_']})
		self.eye_blink_sevice = EyeBlinkService({'face_model': args['face_model']})
		
		# variables

		self.cnt = 0
		self.ALARM_ON = False
		self.use_activity_danger = False
		self.activity_freq = 5
		self.danger_value_threshold = 8

		self.engine = pyttsx3.init()
		self.engine.setProperty('volume', 1.0)

 
	def process(self, frame):

		self.cnt += 1

		# eye blink

		eye_blink_result = self.eye_blink_sevice.process({
			'frame': frame, 
			'ALARM_ON': self.ALARM_ON.
			'danger_value_threshold': danger_value_threshold}
		)
		frame = eye_blink_result['frame']
		ear = eye_blink_result['ear']
		danger_scale_value_ear = eye_blink_result['danger_scale_value']
		NEW_ALARM_ON = eye_blink_result['ALARM_ON']
		sound_text = eye_blink_result['sound_text']

		# activity

		if self.cnt % self.activity_freq == 0:
			activity_result = self.activity_service.process({
				'frame': frame, 
				'ALARM_ON': self.ALARM_ON,
				'danger_value_threshold': self.danger_value_threshold}
			)
			frame = activity_result['frame']
			activity_class = activity_result['activity_class']
			danger_scale_value_activity = activity_result['danger_scale_value']

			if self.use_activity_danger:
				NEW_ALARM_ON |= activity_result['ALARM_ON']
				sound_text = sound_text if not activity_result['ALARM_ON'] else activity_result['sound_text']
		else:
			activity_class = self.activity_service.get_last_activity_class()

		# play sound

		if NEW_ALARM_ON and not self.ALARM_ON:
			self.ALARM_ON = True
			alarm_thread = Thread(target=self.sound_alarm, args=("assets/alarm.wav", sound_text))
			alarm_thread.deamon = True
			alarm_thread.start()

		# danger scale

		if self.use_activity_danger:
			danger_scale_value = (danger_scale_value_ear + danger_scale_value_activity + 1) // 2
			danger_scale_value = max(1, min(10, danger_scale_value))
		else:
			danger_scale_value = danger_scale_value_ear


		return {
			'frame': frame,
			'activity_class': activity_class,
			'ear': ear,
			'danger_scale_value': danger_scale_value
		}


	def sound_alarm(self, path, sound_text):

		playsound.playsound(path)
		self.engine.say(sound_text)
		self.engine.runAndWait()
		playsound.playsound(path)
		self.ALARM_ON = False


	def quit(self):
		self.engine.stop()
 