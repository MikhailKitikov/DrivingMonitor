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

from src.driver_monitor import DriverMonitor

 
class App:

	def __init__(self, window, window_title, activity_model_, face_model, video_source=0):

		# video stream

		self.window = window
		self.window.title(window_title)        
		self.window.bind('<Escape>', lambda e: self.quit())        
		self.video_source = video_source
		self.vid = VideoStream(self.video_source)

		# ui elements

		self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
		self.canvas.pack()

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
		
		# system

		self.driver_monitor = DriverMonitor({'activity_model_': activity_model_, 'face_model': face_model})
		
		# settings

		self.delay = 5
		self.update()
		self.window.protocol("WM_DELETE_WINDOW", self.quit)
		self.window.mainloop()

 
	def update(self):

		# get frame

		ret, frame = self.vid.get_frame()
		if not ret:
			return

		# run monitor

		result = self.driver_monitor.process(frame)
		danger_scale_value = result['danger_scale_value']
		frame = result['frame']
		self.drowsiness_label_text_var.set(self.drowsiness_label_template.format(result['ear']))
		self.activity_label_text_var.set(self.activity_label_template.format(result['activity_class']))
		
		# danger scale

		self.danger_scale_img = PIL.Image.open("assets/images/danger_scale_{}.png".format(danger_scale_value))
		self.danger_scale_img = PIL.ImageTk.PhotoImage(self.danger_scale_img)
		self.danger_scale.configure(image=self.danger_scale_img)

		# show

		self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
		self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
		self.window.after(self.delay, self.update)
		

	def quit(self):
		if self.vid:
			del self.vid
		print("[INFO] Stream closed")  
		self.window.destroy()
		self.driver_monitor.quit()
 

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
