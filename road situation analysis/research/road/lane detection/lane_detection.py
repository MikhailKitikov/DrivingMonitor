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
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

 
class LaneDetectionService:

	def __init__(self, args):

		# build models

		print("[INFO] loading segmentation model...")
		self.unet = load_model("models/road_line_segmentation_unet.hdf5")
		
		# variables

		self.left_fit_list = []
		self.right_fit_list = []
		self.center_fit_list = []

		self.left_fitx_list = []
		self.right_fitx_list = []
		self.center_fitx_list = []

		self.XM_PER_PIX = 3.7 / 720
		self.SRC = np.float32([[690, 440], [790, 440], [560, 680], [1260, 680]])
		self.DST = np.float32([[[200, 0], [1200, 0], [200, 710], [1200, 710]]], dtype=np.int32)


	def segment(self, frame):

		seg = self.unet.predict(np.expand_dims(preprocess_input(frame), axis=0))[0]
		seg = (seg * 255).astype('uint8')

		return seg


	def perspectiveWarp(self, img):

		img_size = (img.shape[1], img.shape[0])
		matrix = cv2.getPerspectiveTransform(self.SRC, self.DST)
		minv = cv2.getPerspectiveTransform(self.DST, self.SRC)
		birdseye = cv2.warpPerspective(img, matrix, img_size)
		height, width = birdseye.shape[:2]

		birdseyeLeft  = birdseye[0:height, 0:width // 2]
		birdseyeRight = birdseye[0:height, width // 2:width]

		return birdseye, birdseyeLeft, birdseyeRight, minv


	def plotHistogram(self, img):

		histogram = np.sum(img[img.shape[0] // 2:, :], axis = 0)

		midpoint = np.int(histogram.shape[0] / 2)
		leftxBase = np.argmax(histogram[:midpoint])
		rightxBase = np.argmax(histogram[midpoint:]) + midpoint

		return histogram, leftxBase, rightxBase


	def slide_window_search(self, binary_warped, histogram):

		out_img = np.dstack((binary_warped, binary_warped, binary_warped))
		midpoint = np.int(histogram.shape[0] / 2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		nwindows = 10
		window_height = np.int(binary_warped.shape[0] / nwindows)
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		leftx_current = leftx_base
		rightx_current = rightx_base
		margin = 100
		minpix = 50
		left_lane_inds = []
		right_lane_inds = []

		for window in range(nwindows):
			win_y_low = binary_warped.shape[0] - (window + 1) * window_height
			win_y_high = binary_warped.shape[0] - window * window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
			color_left = (255,255,255) if len(good_left_inds) > 100 else (255,0,0)
			color_right = (255,255,255) if len(good_right_inds) > 100 else (255,0,0)
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)

			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		center_fit = (left_fit + right_fit) / 2

		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
		left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
		right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
		center_fitx = center_fit[0] * ploty**2 + center_fit[1] * ploty + center_fit[2]

		ltx = np.trunc(left_fitx)
		rtx = np.trunc(right_fitx)
		ctx = np.trunc(center_fitx)

		return ploty, left_fit, right_fit, center_fit, ltx, rtx, ctx


	def general_search(self, binary_warped, left_fit, right_fit):

		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
		left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
		left_fit[1]*nonzeroy + left_fit[2] + margin)))

		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
		right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
		right_fit[1]*nonzeroy + right_fit[2] + margin)))

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

		# visualize

		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
									  ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

		ret = {}
		ret['leftx'] = leftx
		ret['rightx'] = rightx
		ret['left_fitx'] = left_fitx
		ret['right_fitx'] = right_fitx
		ret['ploty'] = ploty

		return ret


	def measure_lane_curvature(self, ploty, leftx, rightx, center_fit):

		leftx = leftx[::-1]
		rightx = rightx[::-1]
		y_eval = np.max(ploty)

		if center_fit[0] < -0.0001:
			curve_direction = 'Left Curve'
		elif center_fit[0] > 0.0001:
			curve_direction = 'Right Curve'
		else:
			curve_direction = 'Straight'

		return None, curve_direction


	def draw_lane_lines(self, original_image, warped_image, Minv, draw_info, 
		points_y_left, points_x_left, points_y_right, points_x_right, points_y_center, points_x_center):

		leftx = draw_info['leftx']
		rightx = draw_info['rightx']
		left_fitx = draw_info['left_fitx']
		right_fitx = draw_info['right_fitx']
		ploty = draw_info['ploty']

		warp_zero = np.zeros_like(warped_image).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		mean_x = np.mean((left_fitx, right_fitx), axis=0)
		pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

		cv2.fillPoly(color_warp, np.int_([pts]), (255, 255, 255))

		points = np.array(list(zip(points_x_left, points_y_left))).astype(int)
		color_warp = cv2.polylines(color_warp, [points], False, (255, 0, 0), 15) 

		points = np.array(list(zip(points_x_right, points_y_right))).astype(int)
		color_warp = cv2.polylines(color_warp, [points], False, (255, 0, 0), 15) 

		points = np.array(list(zip(points_x_center, points_y_center))).astype(int)
		color_warp = cv2.polylines(color_warp, [points], False, (255, 0, 0), 10) 

		newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
		result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

		return pts_mean, result


	def offCenter(self, meanPts, inpFrame):

		mpts = meanPts[-1][-1][-2].astype(int)
		pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
		deviation = pixelDeviation * self.XM_PER_PIX
		direction = "left" if deviation < 0 else "right"

		return deviation, direction


	def addText(self, img, radius, direction, deviation, devDirection):

		img1 = np.copy(img)

		if direction == 'Straight':
			start_point = (1280 // 2, 100)
			end_point = (1280 // 2, 100 - 25)
			color = (255, 0, 0) 
			thickness = 7
			img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness, tipLength = 0.4)
		elif direction == 'Left Curve':
			start_point = (1280 // 2, 100)
			end_point = (1280 // 2 - 25, 100 - 25)
			color = (255, 0, 0) 
			thickness = 7
			img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness, tipLength = 0.4)
		elif direction == 'Right Curve':
			start_point = (1280 // 2, 100)
			end_point = (1280 // 2 + 25, 100 - 25)
			color = (255, 0, 0) 
			thickness = 7
			img1 = cv2.arrowedLine(img1, start_point, end_point, color, thickness, tipLength = 0.4)

		img = cv2.addWeighted(img, 0.7, img1, 0.3, 0)

		deviation_text = 'Deviation: ' + str(round(abs(deviation), 3)) + 'm ' + devDirection
		cv2.putText(img, deviation_text, (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

		return img

 
	def process(self, frame):

		# find lines

		seg = self.segment(frame)
		birdView, _, _, minverse = self.perspectiveWarp(seg)
		hist, _, _ = self.plotHistogram(birdView)
		ploty, left_fit, right_fit, center_fit, left_fitx, right_fitx, center_fitx = self.slide_window_search(birdView, hist)
		draw_info = self.general_search(birdView, left_fit, right_fit)

		# smoothen

		if len(self.left_fit_list):
			left_fit = 0.5 * left_fit + 0.5 * self.left_fit_list[-1]
		if len(self.right_fit_list):
			right_fit = 0.5 * right_fit + 0.5 * self.right_fit_list[-1]
		if len(self.center_fit_list):
			center_fit = 0.5 * center_fit + 0.5 * self.center_fit_list[-1]

		if len(self.left_fit_list) and np.linalg.norm(left_fit - self.left_fit_list[-1]) > 150:
			left_fit = self.left_fit_list[-1]
			left_fitx = self.left_fitx_list[-1]
		self.left_fit_list.append(left_fit)
		self.left_fitx_list.append(left_fitx)

		if len(self.lright_fit_list) and np.linalg.norm(right_fit - self.right_fit_list[-1]) > 150:
			right_fit = right_fit_list[-1]
			right_fitx = right_fitx_list[-1]
		self.right_fit_list.append(right_fit)
		self.right_fitx_list.append(right_fitx)

		if len(self.center_fit_list) and np.linalg.norm(center_fit - self.center_fit_list[-1]) > 150:
			center_fit = self.center_fit_list[-1]
			center_fitx = self.center_fitx_list[-1]
		self.center_fit_list.append(center_fit)
		self.center_fitx_list.append(center_fitx)

		# calculate characteristics

		curveRad, curveDir = self.measure_lane_curvature(ploty, left_fitx, right_fitx, center_fit)
		meanPts, result = self.draw_lane_lines(frame, birdView, minverse, draw_info, 
			ploty, left_fitx, ploty, right_fitx, ploty, center_fitx)

		deviation, directionDev = self.offCenter(meanPts, frame)
		frame = self.addText(result, curveRad, curveDir, deviation, directionDev)

		return frame
