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

from utils.activity import *
from utils.eye_blink import *

 
class App:

	def __init__(self, window, window_title, activity_model_, face_model, video_source=0):

		self.window = window
		self.window.title(window_title)        
		self.window.bind('<Escape>', lambda e: self.quit())        
		self.video_source = video_source

		self.activity_model_ = activity_model_
		self.face_model = face_model

		# open video stream
		self.vid = VideoStream(self.video_source)

		# create canvas
		self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas.pack()

		# info elements
		self.activity_label_text_var = StringVar()
		self.activity_label_template = "Activity: {}"
		self.activity_label_text_var.set(self.activity_label_template.format(' '))
		self.activity_label = Label(self.window, textvariable=self.activity_label_text_var)
		self.activity_label .pack(anchor=SW)

		self.drowsiness_label_text_var = StringVar()
		self.drowsiness_label_template = "Eye ratio: {}%"
		self.drowsiness_label_text_var.set(self.drowsiness_label_template.format(' '))
		self.drowsiness_label = Label(self.window, textvariable=self.drowsiness_label_text_var)
		self.drowsiness_label.pack(anchor=SW)

		self.danger_scale_img = PIL.Image.open("assets/images/danger_scale_1.png")
		self.danger_scale_img = PIL.ImageTk.PhotoImage(self.danger_scale_img)
		self.danger_scale = Label(self.window, image=self.danger_scale_img)
		self.danger_scale.pack(anchor=S)

		# build models
		print("[INFO] loading activity model...")
		if activity_model_ == 'mobilenet-v2-distill':
			self.activity_model = load_mobilenet_v2_distill()
		else:
			raise Exception("Unknown model: {}".format(activity_model))

		print("[INFO] loading face detector...")
		if face_model == 'CNN':
			self.detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
		else:
			self.detector = dlib.get_frontal_face_detector()

		print("[INFO] loading landmark detector...")
		if face_model == 'CNN':
			self.predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat")
		else:
			self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
		
		# variables
		self.cnt = 0
		self.classes = load_classes()

		self.EYE_AR_THRESH = 0.3
		self.EYE_AR_CONSEC_FRAMES = 48
		self.COUNTER = 0
		self.ALARM_ON = False

		self.ear_states = deque(maxlen=50)
		self.activity_states = deque(maxlen=10)

		self.danger_scale_value_ear = 1
		self.danger_scale_value_activity = 1

		self.engine = pyttsx3.init()
		self.engine.setProperty('volume', 1.0)

		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
		
		# settings
		self.delay = 5
		self.update()
		self.window.protocol("WM_DELETE_WINDOW", self.quit)
		self.window.mainloop()

 
	def update(self):    

		ret, frame = self.vid.get_frame()
		if not ret:
			return

		# eye blink

		ear = self.run_eye_blink(frame)
		ear = ear if ear is not None else 0
		self.ear_states.append(int(ear < self.EYE_AR_THRESH))

		if len(self.ear_states) >= self.ear_states.maxlen:

			closed_eyes_frac = np.mean(self.ear_states)
			self.danger_scale_value_ear = max(1, min(10, int(round(closed_eyes_frac * 10))))

			if self.danger_scale_value_ear > 5:
				if not self.ALARM_ON:
					self.ALARM_ON = True
					alarm_thread = Thread(target=self.sound_alarm, args=("assets/alarm.wav",))
					alarm_thread.deamon = True
					alarm_thread.start()
					# cv2.putText(frame, "DROWSINESS ALERT!", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			else:
				self.ALARM_ON = False

		# activity

		use_activity_danger = False
			
		if self.cnt % 5 == 0:

			activity_class = self.run_activity(frame)

			self.activity_label_text_var.set(self.activity_label_template.format(self.classes[activity_class]))
			self.activity_states.append(int(activity_class != 0))

			if len(self.activity_states) >= self.activity_states.maxlen:

				bad_activity_frac = np.mean(self.activity_states)
				self.danger_scale_value_activity = max(1, min(10, int(round(bad_activity_frac * 10))))

				if use_activity_danger:
					if self.danger_scale_value_activity > 5:
						if not self.ALARM_ON:
							self.ALARM_ON = True
							alarm_thread = Thread(target=self.sound_alarm, args=("assets/alarm.wav",))
							alarm_thread.deamon = True
							alarm_thread.start()
							# cv2.putText(frame, "DISTRACTED ALERT!", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					else:
						self.ALARM_ON = False

		# danger scale

		if use_activity_danger:
			danger_scale_value = (self.danger_scale_value_ear + self.danger_scale_value_activity + 1) // 2
			danger_scale_value = max(1, min(10, danger_scale_value))
		else:
			danger_scale_value = self.danger_scale_value_ear

		self.danger_scale_img = PIL.Image.open("assets/images/danger_scale_{}.png".format(danger_scale_value))
		self.danger_scale_img = PIL.ImageTk.PhotoImage(self.danger_scale_img)
		self.danger_scale.configure(image=self.danger_scale_img)

		# show

		self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
		self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

		self.cnt += 1
		self.window.after(self.delay, self.update)


	def sound_alarm(self, path):

		playsound.playsound(path)
		self.engine.say("Please! Watch the road!")
		self.engine.runAndWait()
		playsound.playsound(path)


	def run_activity(self, frame):

		frame = cv2.resize(frame, (224, 224)).astype('float32')
		if self.activity_model_.startswith('mobilenet-v2'):
			frame = preprocess_input(frame)

		pred = self.activity_model(frame[np.newaxis, ...])[0]

		return np.argmax(pred)


	def run_eye_blink(self, frame):

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
		ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2
		self.drowsiness_label_text_var.set(self.drowsiness_label_template.format(round(ear * 100, 2)))
		
		# visualize eyes
		cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

		return ear
		

	def quit(self):
		if self.vid:
			del self.vid
		print("[INFO] Stream closed")  
		self.window.destroy()
		self.engine.stop()
 

class VideoStream:

	def __init__(self, video_source=0):

		# open video
		self.vid = cv2.VideoCapture(video_source)
		if not self.vid.isOpened():
			raise ValueError("[INFO] Unable to open video source", video_source)

		# get dimensions
		self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

		print("[INFO] Stream opened successfully")

 
	def get_frame(self):
		if self.vid.isOpened():
			ret, frame = self.vid.read()
			if ret:
				return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			else:
				return (ret, None)
		else:
			return (ret, None)


	def __del__(self):
		if self.vid.isOpened():
			self.vid.release()


if __name__ == '__main__':

	# start session
	NUM_PARALLEL_EXEC_UNITS = 4
	config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=4,\
						   allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
	session = tf.compat.v1.Session(config=config)
	
	# parse command line arguments    
	try:
		parser = argparse.ArgumentParser()
		parser.add_argument("--stream", help="stream path")
		parser.add_argument("--activity_model", help="activity model name", default='mobilenet-v2-distill')
		parser.add_argument("--face_model", help="face model type", default='CLASSIC')
		args = parser.parse_args()
		stream = 0
		
		if args.stream:
			stream = args.stream
			try:
				stream = int(stream)
			except:
				pass

		print("[INFO] Opening stream " + str(stream) + " ...")

		#  create window
		title = "DriverMonitor (© Mikhail Kitikov)"
		app = App(tkinter.Tk(), title, args.activity_model, args.face_model, stream)   

	except Exception as e:
		print('[INFO] Stream failed (' + str(e) + ')')
