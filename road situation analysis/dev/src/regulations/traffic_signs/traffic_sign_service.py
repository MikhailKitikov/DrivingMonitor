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

from tensorflow.keras.models import load_model

 
class TrafficSignService:

	def __init__(self, args):

		# build models

		print("[INFO] loading traffic sign detection model...")
		self.graph = tf.Graph()
		with graph.as_default():
			graph_def = tf.GraphDef()
			with tf.gfile.GFile("models/regulation_detector_frozen.pb", 'rb') as file:
				graph_def.ParseFromString(file.read())
				tf.import_graph_def(graph_def, name="")

		print("[INFO] loading traffic sign classification model...")
		self.classifier = load_model('models/traffic_sign_classifier.hdf5')

		# variables

		self.traffic_sign_relevant_states = {
			0: 'speed_20',
			1: 'speed_30',
			2: 'speed_50',
			3: 'speed_60',
			4: 'speed_70',
			5: 'speed_80',
			6: 'speed_90',
			14: 'stop',
			27: 'crosswalk'
		}

 
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

				crops = []

				for i in range(len(boxes[0])):
					xmin, ymin = int(boxes[0,0,1] * im_width), int(boxes[0,0,0] * im_height)
					xmax, ymax = int(boxes[0,0,3] * im_width), int(boxes[0,0,2] * im_height)
					cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

					crop = sess.run(tf.image.crop_to_bounding_box(frame, ymin, xmin, ymax - ymin, xmax - xmin))
					crop = self.preprocess_img(crop)
					crops.append(crop)

		# classify

		traffic_signs = []

		for img in crops:
			img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
			pred = self.classifier.predict(np.expand_dims(img, 0))[0]
			if pred in self.traffic_sign_relevant_states:
				traffic_sign = self.traffic_sign_relevant_states[pred]
				print(f"Warning! Traffic sign '{traffic_sign}' detected.")
				traffic_signs.append(traffic_sign)

		return traffic_signs
 