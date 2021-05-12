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

 
class RoadMonitor:

	def __init__(self, args):

		# build models

		print("[INFO] loading tracking model...")
		self.detector = load_centertrack()
		
		# variables

		self.cnt = 0

		self.danger_scale_value_ear = 1
		self.danger_scale_value_activity = 1

 
	def process(self, frame):

		self.cnt += 1

		


		return {
			'frame': frame,
			'activity_class': self.classes[self.activity_states[-1]] if self.activity_states else None,
			'ear': round(ear * 100, 2),
			'danger_scale_value': danger_scale_value
		}
 