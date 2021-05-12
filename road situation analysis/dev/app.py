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

 
class App:

	def __init__(self, window, window_title, video_source=0):

		self.window = window
		self.window.title(window_title)        
		self.window.bind('<Escape>', lambda e: self.quit())        
		self.video_source = video_source

		# open video stream
		self.vid = VideoStream(self.video_source)

		# create canvas
		self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas.pack()

		# radio buttons
		# self.display_mode = IntVar()
  #       Radiobutton(self.window, text="Main", variable=self.display_mode, value=0).pack(anchor=S)
  #       Radiobutton(self.window, text="Depth", variable=self.display_mode, value=1).pack(anchor=S)
  #       Radiobutton(self.window, text="Bird's-eye view", variable=self.display_mode, value=2).pack(anchor=S)

		# info elements
		# self.danger_scale_img = PIL.Image.open("assets/images/danger_scale_1.png")
		# self.danger_scale_img = PIL.ImageTk.PhotoImage(self.danger_scale_img)
		# self.danger_scale = Label(self.window, image=self.danger_scale_img)
		# self.danger_scale.pack(anchor=S)

		# build models
		# print("[INFO] loading activity model...")
		# self.activity_model = load_mobilenet_v2_distill()
		
		# variables
		self.cnt = 0
		# self.classes = load_classes()

		# self.COUNTER = 0
		# self.ALARM_ON = False

		# self.ear_states = deque(maxlen=50)
		# self.activity_states = deque(maxlen=10)

		# self.danger_scale_value_ear = 1
		# self.danger_scale_value_activity = 1

		# self.engine = pyttsx3.init()
		# self.engine.setProperty('volume', 1.0)
		
		# settings
		self.delay = 5
		self.update()
		self.window.protocol("WM_DELETE_WINDOW", self.quit)
		self.window.mainloop()

 
	def update(self):    

		ret, frame = self.vid.get_frame()
		if not ret:
			return

		# danger scale

		# danger_scale_value = self.danger_scale_value_ear
		# self.danger_scale_img = PIL.Image.open("assets/images/danger_scale_{}.png".format(danger_scale_value))
		# self.danger_scale_img = PIL.ImageTk.PhotoImage(self.danger_scale_img)
		# self.danger_scale.configure(image=self.danger_scale_img)

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
		parser.add_argument("--stream_type", help="mono / stereo")
		parser.add_argument("--stream_stereo_left", default=None)
		parser.add_argument("--stream_stereo_right", default=None)
		parser.add_argument("--stream_mono", default=None)
		args = parser.parse_args()

		print(f"[INFO] Opening {args.stream_type} stream...")

		# create window
		title = "RoadMonitor (© Mikhail Kitikov)"
		app = App(tkinter.Tk(), title, args)   

	except Exception as e:
		print('[INFO] Stream failed (' + str(e) + ')')
