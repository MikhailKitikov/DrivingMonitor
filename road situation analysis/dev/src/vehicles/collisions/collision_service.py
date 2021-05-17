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
import pickle
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from collections import deque

 
class CollisionService:

	def __init__(self, args):

		pass


	def find_by_id(self, detection_queue, car_id):

		for obj in detection_queue:
			if obj['id'] == car_id:
				return obs

		return None


	def line_intersect(self, Ax1, Ay1, Ax2, Ay2, Bx1, By1, Bx2, By2):
		""" returns a (x, y) tuple or None if there is no intersection """

		d = (By2 - By1) * (Ax2 - Ax1) - (Bx2 - Bx1) * (Ay2 - Ay1)

		if d:
			uA = ((Bx2 - Bx1) * (Ay1 - By1) - (By2 - By1) * (Ax1 - Bx1)) / d
			uB = ((Ax2 - Ax1) * (Ay1 - By1) - (Ay2 - Ay1) * (Ax1 - Bx1)) / d
		else:
			return None

		if not(0 <= uA <= 1 and 0 <= uB <= 1):
			return

		x = Ax1 + uA * (Ax2 - Ax1)
		y = Ay1 + uA * (Ay2 - Ay1)
	 
		return x, y


	def point_dist(self, A, B):
		return np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

 
	def process(self, detection_queue, current_car_ids, dt, max_speed):

		for car_id in current_car_ids:

			if len(detection_queue) < 4:
				continue

			# get car states

			car_state_time_1 = self.find_by_id(detection_queue[-4], car_id)
			car_state_time_2 = self.find_by_id(detection_queue[-3], car_id)
			car_state_time_3 = self.find_by_id(detection_queue[-2], car_id)
			car_state_time_4 = self.find_by_id(detection_queue[-1], car_id)

			# filter cars in radius 20m

			dist_to_car = np.sqrt(car_state_time_4['center'][0] ** 2 + car_state_time_4['center'][1] ** 2)

			if dist_to_car > 20:
				continue

			# calculate position shift vectors

			pos_shift_x_12 = car_state_time_2['center'][0] - car_state_time_1['center'][0]
			pos_shift_y_12 = car_state_time_2['center'][1] - car_state_time_1['center'][1]

			pos_shift_x_23 = car_state_time_3['center'][0] - car_state_time_2['center'][0]
			pos_shift_y_23 = car_state_time_3['center'][1] - car_state_time_2['center'][1]

			pos_shift_x_34 = car_state_time_4['center'][0] - car_state_time_3['center'][0]
			pos_shift_y_34 = car_state_time_4['center'][1] - car_state_time_3['center'][1]

			# calculate velocity vectors

			velocity_x_12, velocity_y_12 = pos_shift_x_12 / dt, pos_shift_y_12 / dt
			velocity_x_23, velocity_y_23 = pos_shift_x_23 / dt, pos_shift_y_23 / dt
			velocity_x_34, velocity_y_34 = pos_shift_x_34 / dt, pos_shift_y_34 / dt

			# calculate weighted sum

			agg_velocity_x = velocity_x_12  / (2 ** 3) + velocity_x_23  / (2 ** 2) + velocity_x_34  / (2 ** 1)
			agg_velocity_y = velocity_y_12  / (2 ** 3) + velocity_y_23  / (2 ** 2) + velocity_y_34  / (2 ** 1)

			agg_shift_x, agg_shift_y = agg_velocity_x * dt, agg_velocity_y * dt

			# define danger zone boundaries

			TOP_RIGHT_CORNER_X, TOP_RIGHT_CORNER_Y = 2.5, 4
			TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y = -2.5, 4
			BOTTOM_RIGHT_CORNER_X, BOTTOM_RIGHT_CORNER_Y = 2.5, -4
			BOTTOM_LEFT_CORNER_X, BOTTOM_LEFT_CORNER_Y = -2.5, -4

			# define car movement vector

			car_start_point_x, car_start_point_y = car_state_time_4['center'][0], car_state_time_4['center'][1]
			car_end_point_x, car_end_point_y = car_start_point_x + agg_shift_x, car_start_point_y + agg_shift_y
			car_curr_position = (car_start_point_x, car_start_point_y)

			# calculate intersections

			right_boundary_intersection = self.line_intersect(
				BOTTOM_RIGHT_CORNER_X, BOTTOM_RIGHT_CORNER_Y, TOP_RIGHT_CORNER_X, TOP_RIGHT_CORNER_Y,
				car_start_point_x, car_start_point_y, car_end_point_x, car_end_point_y)

			left_boundary_intersection = self.line_intersect(
				BOTTOM_LEFT_CORNER_X, BOTTOM_LEFT_CORNER_Y, TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y,
				car_start_point_x, car_start_point_y, car_end_point_x, car_end_point_y)

			top_boundary_intersection = self.line_intersect(
				TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y, TOP_RIGHT_CORNER_X, TOP_RIGHT_CORNER_Y,
				car_start_point_x, car_start_point_y, car_end_point_x, car_end_point_y)

			intersection_candidates = []
			for cand in [right_boundary_intersection, left_boundary_intersection, top_boundary_intersection]:
				if cand is None:
					continue
				intersection_candidates.append(cand)

			if len(intersection_candidates) == 0:
				continue

			intersection_candidates = sorted(intersection_candidates, key=lambda cand: self.point_dist(cand, car_curr_position))
			intersection = intersection_candidates[0]

			# filter cars with < 5 sec to collision

			speed_along_vec = np.sqrt(agg_velocity_x ** 2 + agg_velocity_y ** 2)
			time_to_collision = self.point_dist(intersection, car_curr_position) / speed_along_vec

			if time_to_collision > 5:
				continue

			# calculate angle

			angle = 90
			if intersection[0] != car_curr_position[0]:
				slope = (intersection[1] - car_curr_position[1]) / (intersection[0] - car_curr_position[0])
				angle = np.arctan([slope])[0] * 180 / np.pi

			print(f"Warning! Collision possible in {time_to_collision} sec from angle {angle} deg.")
