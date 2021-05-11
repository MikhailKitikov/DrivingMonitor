import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
import pytesseract

from my_detection_utils import *

tf.disable_v2_behavior()


# define params
CKPT_PATH = "models/ckpt/frozen_inference_graph.pb"
IMAGE_DIR = 'images/'


# create graph
graph = tf.Graph()

with graph.as_default():
	graph_def = tf.GraphDef()
	with tf.gfile.GFile(CKPT_PATH, 'rb') as file:
		graph_def.ParseFromString(file.read())
		tf.import_graph_def(graph_def, name="")


# init tesseract
pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# detect
with graph.as_default():

	with tf.Session(graph=graph) as sess:

		image_tensor = graph.get_tensor_by_name('image_tensor:0')
		detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
		detection_scores = graph.get_tensor_by_name('detection_scores:0')
		detection_classes = graph.get_tensor_by_name('detection_classes:0')
		num_detections = graph.get_tensor_by_name('num_detections:0')

		for image_path in os.listdir(IMAGE_DIR):
			image_path = os.path.join(IMAGE_DIR, image_path)
			image = cv2.imread(image_path)
			image_exp = image[None, ...]
			im_height, im_width = image.shape[:2]

			boxes, scores, classes, num = sess.run(
			  [detection_boxes, detection_scores, detection_classes, num_detections],
			  feed_dict={image_tensor: image_exp})

			xmin, ymin = int(boxes[0,0,1] * im_width), int(boxes[0,0,0] * im_height)
			xmax, ymax = int(boxes[0,0,3] * im_width), int(boxes[0,0,2] * im_height)

			img = sess.run(tf.image.crop_to_bounding_box(image, ymin, xmin, ymax - ymin, xmax - xmin))
			img = preprocess_img(img)

			text = pytesseract.image_to_string(
				img, config ='--oem 3 -l eng --psm 6')
			text = verify_text(text)
			print(f'Plate number: {text}')
