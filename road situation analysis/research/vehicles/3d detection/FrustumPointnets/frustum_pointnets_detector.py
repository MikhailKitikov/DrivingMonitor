import warnings
warnings.filterwarnings("ignore")

import os
import sys
import glob
from tqdm import tqdm
from collections import namedtuple
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('frustum_pointnets\\train'))
sys.path.insert(0, os.path.abspath('frustum_pointnets\\train\\test_mod'))
sys.path.insert(0, os.path.abspath('frustum_pointnets\\kitti'))

from ssd import SSD
import provider
from test_mod import inference, get_session_and_ops
from kitti_util import Calibration
from kitti_object import get_lidar_in_image_fov, kitti_object


Detection = namedtuple('Detection', ['xyz', 'angle', 'lwh', 'confidence'])
Scene = namedtuple('Scene', ['detections'])

class FrustumPointnetsDetector(object):

	def __init__(self, ssd_detector, ssd_threshold, frustum_pointnet, frustum_batch_size, frustum_num_pts):
		
		self.ssd_detector = ssd_detector
		self.ssd_threshold = ssd_threshold
		
		self.frustum_pointnet = frustum_pointnet
		self.frustum_batch_size = frustum_batch_size
		self.frustum_num_pts = frustum_num_pts
		self.frustum_sess, self.frustum_ops = get_session_and_ops(self.frustum_batch_size, self.frustum_num_pts)

  
	def predict(self, velo_pts, image, calib):

		detection = self.ssd_detector.predict(image)
		vehicle_idx = np.where(detection['detection_classes'] == 1)
		conf_idx = np.where(detection['detection_scores'] >= self.ssd_threshold)
		final_idx = np.intersect1d(vehicle_idx, conf_idx)
		bbox = detection['detection_boxes'][final_idx]    
		detection_conf = detection['detection_scores'][final_idx]

		rect_pts = np.zeros_like(velo_pts)
		rect_pts[:, :3] = calib.project_velo_to_rect(velo_pts[:, :3].copy())
		rect_pts[:, 3] = velo_pts[:, 3]
		
		img_height, img_width, _= image.shape
		_, img_pts, in_img_mask = get_lidar_in_image_fov(velo_pts[:, :3].copy(), calib, 0, 0, img_width, img_height, return_more=True)
		 
		frustum_examples = []
		frustum_angles = []
		scene = Scene([])

		for box in bbox:
			box = (box.reshape((2, 2)) * image.shape[:2]).astype(int)
			(ul_y, ul_x), (lr_y, lr_x) = box
			box_mask = (img_pts[:, 1] < lr_y) * (img_pts[:, 1] >= ul_y) * (img_pts[:, 0] < lr_x) * (img_pts[:, 0] >= ul_x) 
			
			mask = in_img_mask & box_mask
			rect_pts_masked = rect_pts[mask]            

			if len(rect_pts_masked):
				box2d_center = np.array([(ul_x + lr_x) / 2.0, (ul_y + ul_y) / 2.0])
				uvdepth = np.zeros((1, 3))
				uvdepth[0, :2], uvdepth[0, 2] = box2d_center, 10
				box2d_center_rect = calib.project_image_to_rect(uvdepth)
				frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2], box2d_center_rect[0, 0])
				frustum_angle += np.pi / 2.0
				
				np.random.seed()
				point_cloud = provider.rotate_pc_along_y(rect_pts_masked.copy(), frustum_angle)
				idx = np.random.choice(len(point_cloud), size=self.frustum_num_pts, replace=True)
				point_cloud = point_cloud[idx]
				
				frustum_angles.append(frustum_angle)
				frustum_examples.append(point_cloud)
			
		if len(frustum_examples):
			one_hot_batch = np.array([[1, 0, 0],] * len(frustum_examples)).reshape(-1, 3)        

			predictions = self.frustum_pointnet(
				self.frustum_sess, self.frustum_ops, np.array(frustum_examples), one_hot_batch, self.frustum_batch_size)
			_, centers, heading_cls, heading_res, size_cls, size_res, _ = predictions

			for i, _ in enumerate(heading_cls):  
				h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(
					centers[i], heading_cls[i], heading_res[i], size_cls[i], size_res[i], frustum_angles[i])
				detection = Detection(xyz=np.array((tx, ty, tz)), angle=ry, lwh=np.array((l, w, h)), confidence=detection_conf[i])
				scene.detections.append(detection)

			return scene


if __name__ == '__main__':

	ssd = SSD('')
	detector = FrustumPointnetsDetector(
		ssd_detector=ssd, ssd_threshold=0.3, frustum_pointnet=inference, frustum_batch_size=1, frustum_num_pts=256)

	dataset = kitti_object(root_dir='kitti_hw_dataset')

	idx = 8
	img = dataset.get_image(idx)
	lidar = dataset.get_lidar(idx)
	calib = dataset.get_calibration(idx)
	labels = dataset.get_label_objects(idx)

	result = detector.predict(lidar, img, calib)
	print(result)
