import tkinter
from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import time
import argparse
import os
from keras import backend as K
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

import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pytesseract

 
class NumberPlateRecognitionService:

	def __init__(self, args):

		# build models

		print("[INFO] loading number plate recognition model...")
		
		self.graph = tf.Graph()
		with graph.as_default():
			graph_def = tf.GraphDef()
			with tf.gfile.GFile("models/number_plate_detector_frozen.pb", 'rb') as file:
				graph_def.ParseFromString(file.read())
				tf.import_graph_def(graph_def, name="")

		pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

 
	def process(self, frame):

		# detect

		with self.graph.as_default():
			with tf.Session(graph=self.graph) as sess:
				image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
				detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
				detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
				detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
				num_detections = self.graph.get_tensor_by_name('num_detections:0')

				image_exp = frame[None, ...]
				im_height, im_width = frame.shape[:2]

				boxes, scores, classes, num = sess.run(
				  [detection_boxes, detection_scores, detection_classes, num_detections],
				  feed_dict={image_tensor: image_exp})

		# recognize

		xmin, ymin = int(boxes[0,0,1] * im_width), int(boxes[0,0,0] * im_height)
		xmax, ymax = int(boxes[0,0,3] * im_width), int(boxes[0,0,2] * im_height)
		cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

		crop = sess.run(tf.image.crop_to_bounding_box(frame, ymin, xmin, ymax - ymin, xmax - xmin))
		crop = self.preprocess_img(crop)

		text = pytesseract.image_to_string(crop, config ='--oem 3 -l eng --psm 6')
		text = self.verify_text(text)

		font = cv2.FONT_HERSHEY_SIMPLEX
		org = (50, 50)
		fontScale = 1
		color = (255, 0, 0)
		thickness = 1
		
		frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

		return {
			'frame': frame,
			'number': text,
		}


	def preprocess_img(self, img):

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (3, 3), 0)

		return img


	def verify_char(self, ch):

		alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
		
		return '' if ch not in alphabet else ch


	def verify_text(self, text):
		return ''.join([self.verify_char(ch) for ch in text])
