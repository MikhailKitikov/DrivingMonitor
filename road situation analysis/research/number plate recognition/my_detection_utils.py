import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
import pytesseract


def preprocess_img(img):

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (3, 3), 0)

	return img


def verify_char(ch):

	alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
	
	return '' if ch not in alphabet else ch


def verify_text(text):
	return ''.join([verify_char(ch) for ch in text])

