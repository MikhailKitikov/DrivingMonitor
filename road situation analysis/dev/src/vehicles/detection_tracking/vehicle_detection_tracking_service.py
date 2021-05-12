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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import json
import copy
import numpy as np

import sys
sys.path.insert(0, "src/dependencies/CenterTrack/src")
sys.path.insert(0, "src/dependencies/CenterTrack/src/lib")
sys.path.insert(0, "src/dependencies/CenterTrack/src/lib/model/networks/DCNv2")

import _init_paths
from opts import opts
from detector import Detector

 
class DetectionService:

	def __init__(self, args):
		
		# build model

		print("[INFO] loading tracking model...")
		self.MODEL_PATH = "models/nuScenes_3Dtracking.pth"
  		self.TASK = "tracking,ddd"
  		self.opt = opts().init("{} --load_model {}".format(TASK, MODEL_PATH).split(' '))
  		self.detector = Detector(opt)

  		# variables

  		self.cnt = 0
  		self.class_mapping = {
  			'car': [0, 1, 2, 3, 4, 6, 7],
  			'pedestrian': [5]
  		}

 
	def process(self, frame):

		self.cnt += 1

		results = self.detector.run(frame)['results']
		filtered_results = {'car': [], 'pedestrian': []}

		for det in results:
			if det['class'] in self.class_mapping['car']:
				filtered_results['car'].append(self.postprocess_detection(det))
			else:
				filtered_results['pedestrian'].append(self.postprocess_detection(det))

		return filtered_results


	def postprocess_detection(self, det):

		return {
			'id': det['tracking_id'],
			'center': det['loc']
		}
 