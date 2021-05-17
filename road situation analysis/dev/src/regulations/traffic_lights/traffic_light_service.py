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

 
class TrafficLightService:

	def __init__(self, args):

		# build models

		print("[INFO] loading traffic light detection model...")
		self.graph = tf.Graph()
		with graph.as_default():
			graph_def = tf.GraphDef()
			with tf.gfile.GFile("models/regulation_detector_frozen.pb", 'rb') as file:
				graph_def.ParseFromString(file.read())
				tf.import_graph_def(graph_def, name="")

		print("[INFO] loading traffic light classification model...")
		self.classifier = load_model('models/traffic_light_classifier.hdf5')

		# variables

		self.traffic_light_colors = ['green', 'yellow', 'red', 'green']

 
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
				
				crops = list(sorted(crops, key=lambda crop: - crop.shape[0] * crop.shape[1]))
				img = crops[0] if len(crops) else None

		# classify

		if img is None:
			traffic_light_color = 'green'
		else:
			img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
			pred = self.classifier.predict(np.expand_dims(image, 0))[0]
			traffic_light_color = self.traffic_light_colors[pred]

		return traffic_light_color
 