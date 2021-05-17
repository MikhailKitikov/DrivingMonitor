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

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from IPython import display
import time
from matplotlib.lines import Line2D

from src.kalman.timestamp import Timestamp
from src.kalman.car import Car
from src.kalman.linear_movement_model import LinearMovementModel
from src.kalman.cycloid_movement_model import CycloidMovementModel
from src.kalman.can_sensor import CanSensor
from src.kalman.gps_sensor import GpsSensor
from src.kalman.imu_sensor import ImuSensor

from src.kalman.car_plotter import CarPlotter
from src.kalman.kalman_car import KalmanCar
from src.kalman.kalman_can_sensor import KalmanCanSensor
from src.kalman.kalman_gps_sensor import KalmanGpsSensor
from src.kalman.kalman_imu_sensor import KalmanImuSensor
from src.kalman.kalman_movement_model import KalmanMovementModel
from src.kalman.kalman_filter import kalman_transit_covariance, kalman_process_observation

 
class StateEstimationService:

	def __init__(self):

		# build models

		print("[INFO] loading kalman car model...")

		self.car = self.create_car(initial_omega=0.05)
		self.kalman_car = self.create_kalman_car(car)

 
	def process(self, frame, dt):

		self.car.move(dt)
		self.kalman_car.move(dt)

		return {
			'position_x': self.kalman_car._position_x,
			'position_y': self.kalman_car._position_y,
			'velocity_x': self.kalman_car._velocity_x,
			'velocity_y': self.kalman_car._velocity_y
		}


	def create_car(
		self,
		initial_position=[5, 5],
		initial_velocity=5,
		initial_omega=0.0,
		initial_yaw=np.pi / 4,
		can_noise_variances=[0.25],
		gps_noise_variances=[1, 1],
		imu_noise_variances=None,
		random_state=0):

		car = Car(
			initial_position=initial_position,
			initial_velocity=initial_velocity,
			initial_yaw=initial_yaw,
			initial_omega=initial_omega
		)

		if can_noise_variances is not None:
			car.add_sensor(CanSensor(noise_variances=can_noise_variances, random_state=random_state))
			random_state += 1
		if gps_noise_variances is not None:
			car.add_sensor(GpsSensor(noise_variances=gps_noise_variances, random_state=random_state))
			random_state += 1
		if imu_noise_variances is not None:
			car.add_sensor(ImuSensor(noise_variances=imu_noise_variances, random_state=random_state))
			random_state += 1

		movement_model = LinearMovementModel()
		car.set_movement_model(movement_model)

		return car


	 def create_kalman_car(self, car, gps_variances=None, can_variances=None, imu_variances=None):

		noise_covariance_density = np.diag([0.1] * 5)

		kalman_car = KalmanCar(
			initial_position=car.initial_position,
			initial_velocity=car.initial_velocity,
			initial_yaw=car.initial_yaw,
			initial_omega=car.initial_omega
		)

		kalman_car.covariance_matrix = noise_covariance_density
		kalman_movement_model = KalmanMovementModel(noise_covariance_density=noise_covariance_density)
		kalman_car.set_movement_model(kalman_movement_model)

		for sensor in car.sensors:
			noise_variances = sensor._noise_variances
			if isinstance(sensor, GpsSensor):
				noise_variances = noise_variances if gps_variances is None else gps_variances
				kalman_sensor = KalmanGpsSensor(noise_variances=noise_variances)
			elif isinstance(sensor, CanSensor):
				noise_variances = noise_variances if can_variances is None else can_variances
				kalman_sensor = KalmanCanSensor(noise_variances=noise_variances)
			elif isinstance(sensor, ImuSensor):
				noise_variances = noise_variances if imu_variances is None else imu_variances
				kalman_sensor = KalmanImuSensor(noise_variances=noise_variances)
			else:
				assert False
			kalman_car.add_sensor(kalman_sensor)

		return kalman_car
